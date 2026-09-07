//! Fixed-cardinality expert routing for the experimental DSV4 matrix path.
//! Both arms retain ascending original slot order inside each expert group.
use crate::dsv4_ep::EpCompute;
use crate::dsv4_ffi;
use crate::mmq_ffi::{
    memra_bind_device, memra_moe_kq_gemm_sk, memra_moe_kq_gemm_sk_gu,
    memra_moe_kq_gemm_sk_gu_half2, memra_moe_kq_gemm_sk_gu_m1, memra_moe_kq_gemm_sk_gu_m1_half2,
    memra_moe_kq_gemm_sk_m1, memra_moe_kq_gemm_sk_m1_half2,
};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use memra_runtime::Gpu;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

type Res<T> = Result<T, String>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GuLaunchKind {
    Scalar,
    M1,
    Half2,
    M1Half2,
}

/// Select the already-gated GU visitor. The conjunction is deliberately a
/// process-local composition of existing doors: it does not create a new
/// environment/default arm, and callers pass `fuse_gu=false` for every
/// batched or non-plain transaction.
fn select_gu_launch(fuse_gu: bool, m1: bool, half2: bool) -> Option<GuLaunchKind> {
    if !fuse_gu {
        return None;
    }
    Some(match (m1, half2) {
        (true, true) => GuLaunchKind::M1Half2,
        (true, false) => GuLaunchKind::M1,
        (false, true) => GuLaunchKind::Half2,
        (false, false) => GuLaunchKind::Scalar,
    })
}

fn gu_receipt_features(kind: GuLaunchKind) -> (bool, bool) {
    (
        matches!(kind, GuLaunchKind::M1 | GuLaunchKind::M1Half2),
        matches!(kind, GuLaunchKind::Half2 | GuLaunchKind::M1Half2),
    )
}

static SPLITK_COMPONENT_SEEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(crate) fn reset_splitk_component_token() {
    SPLITK_COMPONENT_SEEN.store(0, Ordering::Release);
}
fn splitk_component_claim(gpu: &Gpu, gu: bool) -> bool {
    if crate::MOE_M1_SPLITK_COMPONENT.load(Ordering::Acquire) == 0 {
        return false;
    }
    let bit = 1u64 << (gpu.ctx.ordinal() * 2 + usize::from(gu));
    SPLITK_COMPONENT_SEEN.fetch_or(bit, Ordering::AcqRel) & bit == 0
}

static MIRROR_VALIDATE: AtomicBool = AtomicBool::new(true);
static ROUTE_VALIDATE: AtomicBool = AtomicBool::new(true);

pub(crate) fn set_mirror_validation_for_gate(enabled: bool) -> bool {
    MIRROR_VALIDATE.swap(enabled, Ordering::SeqCst)
}

pub(crate) fn mirror_validation_enabled() -> bool {
    MIRROR_VALIDATE.load(Ordering::SeqCst)
}

pub(crate) fn route_validation_enabled() -> bool {
    ROUTE_VALIDATE.load(Ordering::SeqCst)
}

pub(crate) fn set_route_validation_for_gate(enabled: bool) -> bool {
    ROUTE_VALIDATE.swap(enabled, Ordering::SeqCst)
}

pub(crate) fn resolve_program(raw: Option<&str>) -> Res<bool> {
    match raw {
        None | Some("reference") => Ok(false),
        Some("matrix") => Ok(true),
        Some(other) => Err(format!(
            "MEMRA_DSV4_MOE_PROGRAM '{other}' unknown (reference | matrix)"
        )),
    }
}

pub(crate) fn ensure_program(runtime_matrix: bool, state_matrix: bool) -> Res<()> {
    if runtime_matrix != state_matrix {
        return Err("DSV4 matrix/reference state program mismatch".into());
    }
    Ok(())
}

pub(crate) fn resolve(raw: Option<&str>) -> Res<bool> {
    match raw {
        None | Some("host") => Ok(false),
        Some("device") => Ok(true),
        Some(other) => Err(format!(
            "MEMRA_DSV4_GROUPED_ROUTE '{other}' unknown (host | device)"
        )),
    }
}

fn mirror_bytes(rows: usize, cols: usize) -> Res<usize> {
    if rows == 0
        || rows > i32::MAX as usize
        || cols < 128
        || cols > i32::MAX as usize
        || !cols.is_multiple_of(128)
    {
        return Err("invalid FP8 half mirror dimensions".into());
    }
    rows.checked_mul(
        cols.checked_mul(2)
            .and_then(|n| n.checked_add(8))
            .ok_or("FP8 mirror row byte overflow")?,
    )
    .ok_or_else(|| "FP8 mirror byte overflow".into())
}

/// Stable addresses for a lossless FP8-QAT mirror. Validation remains explicit;
/// this storage change alone does not make the grouped path capture-safe.
pub(crate) struct HalfMirror {
    pub half: CudaSlice<u8>,
    pub scale: CudaSlice<f32>,
    status: CudaSlice<i32>,
    host_status: Vec<i32>,
    rows: usize,
    cols: usize,
}

impl HalfMirror {
    pub fn new(s: &Arc<CudaStream>, rows: usize, cols: usize) -> Res<Self> {
        mirror_bytes(rows, cols)?;
        Ok(Self {
            half: s
                .alloc_zeros::<u8>(rows * cols * 2)
                .map_err(|e| format!("FP8 mirror allocation: {e}"))?,
            scale: s
                .alloc_zeros::<f32>(rows)
                .map_err(|e| format!("FP8 mirror scales: {e}"))?,
            status: s
                .alloc_zeros::<i32>(rows)
                .map_err(|e| format!("FP8 mirror status: {e}"))?,
            host_status: vec![0; rows],
            rows,
            cols,
        })
    }

    /// `row_ids`, when present, comes from successfully validated GroupedRoutes;
    /// its token ids are in the source's admitted row range by construction.
    pub fn gather(
        &mut self,
        s: &Arc<CudaStream>,
        codes: &CudaSlice<u8>,
        scales: &CudaSlice<f32>,
        row_ids: Option<&CudaSlice<i32>>,
        rows: usize,
    ) -> Res<()> {
        if rows == 0
            || rows > self.rows
            || codes.is_empty()
            || !codes.len().is_multiple_of(self.cols)
            || codes.len() / 128 != scales.len()
            || row_ids.is_some_and(|ids| ids.len() < rows)
            || (row_ids.is_none() && codes.len() / self.cols < rows)
        {
            return Err("FP8 half mirror input/workspace shape mismatch".into());
        }
        let rc = unsafe {
            dsv4_ffi::memra_dsv4_fp8_gather_half(
                codes.device_ptr(s).0 as *const std::ffi::c_void,
                scales.device_ptr(s).0 as *const f32,
                row_ids.map_or(std::ptr::null(), |ids| ids.device_ptr(s).0 as *const i32),
                self.half.device_ptr_mut(s).0 as *mut std::ffi::c_void,
                self.scale.device_ptr_mut(s).0 as *mut f32,
                self.status.device_ptr_mut(s).0 as *mut i32,
                rows as i32,
                self.cols as i32,
                s.cu_stream() as *mut std::ffi::c_void,
            )
        };
        if rc != 0 {
            return Err(format!("FP8 half mirror kernel rc={rc}"));
        }
        if mirror_validation_enabled() {
            s.memcpy_dtoh(&self.status.slice(..rows), &mut self.host_status[..rows])
                .map_err(|e| format!("FP8 mirror status read: {e}"))?;
            s.synchronize()
                .map_err(|e| format!("FP8 mirror status sync: {e}"))?;
            if let Some(row) = self.host_status[..rows].iter().position(|&v| v != 0) {
                return Err(format!(
                    "FP8-QAT half mirror is not lossless at gathered row {row}, cols={}",
                    self.cols
                ));
            }
        }
        Ok(())
    }
}

pub(crate) struct GroupedWork {
    pub routes: GroupedRoutes,
    pub input: HalfMirror,
    pub intermediate: HalfMirror,
    pub contribution: CudaSlice<f32>,
    pub bytes: u64,
    /// Split-bank TP/EP work keeps the half-intermediate layout private to the grouped
    /// consumer. The surrounding EpCompute remains full-inter sized so the existing
    /// rank-local walk and reduction ABI do not change.
    split_scratch: Option<SplitScratch>,
    plain_single: bool,
    gu_fuse: bool,
    phase: MatrixPhase,
    splitk_scratch: Option<CudaSlice<f32>>,
}

struct SplitScratch {
    h: CudaSlice<f32>,
    hq: CudaSlice<u8>,
    hs: CudaSlice<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MatrixPhase {
    Idle,
    Prepared,
    UpQueued,
    Failed,
}

pub(crate) fn bind_matrix(gpu: &Gpu) -> Res<()> {
    gpu.ctx
        .bind_to_thread()
        .map_err(|e| format!("matrix bind: {e}"))?;
    let rc = unsafe { memra_bind_device(gpu.ctx.ordinal() as i32) };
    if rc != 0 {
        return Err(format!("matrix runtime bind rc={rc}"));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn modelopt_pointers(
    w: u64,
    sc: u64,
    wlen: usize,
    slen: usize,
    experts: usize,
    hidden: usize,
    inter: usize,
) -> Res<Vec<u64>> {
    if experts == 0
        || experts > 512
        || hidden == 0
        || inter == 0
        || !hidden.is_multiple_of(128)
        || !inter.is_multiple_of(128)
    {
        return Err("invalid matrix expert-bank dimensions".into());
    }
    let area = hidden
        .checked_mul(inter)
        .ok_or("matrix bank dimension overflow")?;
    let wb = area / 2;
    let sb = area / 16;
    if experts.checked_mul(3).and_then(|n| n.checked_mul(wb)) != Some(wlen)
        || experts.checked_mul(3).and_then(|n| n.checked_mul(sb)) != Some(slen)
    {
        return Err("matrix expert-bank length mismatch".into());
    }
    w.checked_add(wlen as u64)
        .ok_or("matrix weight address overflow")?;
    sc.checked_add(slen as u64)
        .ok_or("matrix scale address overflow")?;
    let mut result = vec![0; experts * 6];
    for expert in 0..experts {
        for projection in 0..3 {
            result[2 * projection * experts + expert] = w + ((expert * 3 + projection) * wb) as u64;
            result[(2 * projection + 1) * experts + expert] =
                sc + ((expert * 3 + projection) * sb) as u64;
        }
    }
    Ok(result)
}

pub(crate) fn modelopt_table(
    s: &Arc<CudaStream>,
    w: &CudaSlice<u8>,
    sc: &CudaSlice<u8>,
    experts: usize,
    hidden: usize,
    inter: usize,
) -> Res<CudaSlice<u64>> {
    let pointers = modelopt_pointers(
        w.device_ptr(s).0,
        sc.device_ptr(s).0,
        w.len(),
        sc.len(),
        experts,
        hidden,
        inter,
    )?;
    s.clone_htod(&pointers)
        .map_err(|e| format!("matrix bank table upload: {e}"))
}

impl GroupedWork {
    pub fn new(
        s: &Arc<CudaStream>,
        experts: usize,
        slots: usize,
        hidden: usize,
        inter: usize,
    ) -> Res<Self> {
        Self::new_partition(s, experts, 0, experts, slots, hidden, inter)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_partition(
        s: &Arc<CudaStream>,
        global: usize,
        first: usize,
        experts: usize,
        slots: usize,
        hidden: usize,
        inter: usize,
    ) -> Res<Self> {
        // Validate all byte arithmetic before allocating any device plane.
        let input_bytes = mirror_bytes(slots, hidden)?;
        let intermediate_bytes = mirror_bytes(slots, inter)?;
        let contribution_len = slots
            .checked_mul(hidden)
            .ok_or("grouped contribution size overflow")?;
        let extra_bytes = contribution_len
            .checked_mul(4)
            .and_then(|n| n.checked_add(input_bytes))
            .and_then(|n| n.checked_add(intermediate_bytes))
            .ok_or("grouped workspace byte overflow")?;
        let routes = GroupedRoutes::new_partition(s, global, first, experts, slots)?;
        let bytes = routes
            .bytes
            .checked_add(extra_bytes as u64)
            .ok_or("grouped workspace byte overflow")?;
        Ok(Self {
            routes,
            input: HalfMirror::new(s, slots, hidden)?,
            intermediate: HalfMirror::new(s, slots, inter)?,
            contribution: s
                .alloc_zeros::<f32>(contribution_len)
                .map_err(|e| format!("grouped contribution allocation: {e}"))?,
            bytes,
            split_scratch: None,
            plain_single: false,
            gu_fuse: false,
            phase: MatrixPhase::Idle,
        })
    }

    /// Rank-local all-layer TP/EP work. Every rank receives the full global route domain and
    /// every selected expert id remains unchanged; only the ModelOpt bank's intermediate width
    /// is local (`inter/2`). The down projection still emits `hidden` values, which are the
    /// rank partials consumed by the caller's named rank-order reduction.
    pub fn new_split(
        s: &Arc<CudaStream>,
        global: usize,
        slots: usize,
        hidden: usize,
        local_inter: usize,
    ) -> Res<Self> {
        if global == 0 || global > 512 || slots == 0 || local_inter == 0 {
            return Err("invalid split grouped dimensions".into());
        }
        let input_bytes = mirror_bytes(slots, hidden)?;
        let intermediate_bytes = mirror_bytes(slots, local_inter)?;
        let contribution_len = slots
            .checked_mul(hidden)
            .ok_or("split grouped contribution size overflow")?;
        let split_h_len = slots
            .checked_mul(local_inter)
            .ok_or("split grouped intermediate size overflow")?;
        let split_hs_len = split_h_len
            .checked_div(128)
            .ok_or("split grouped scale size overflow")?;
        let split_extra_bytes = split_h_len
            .checked_mul(4)
            .and_then(|n| n.checked_add(split_h_len))
            .and_then(|n| {
                split_hs_len
                    .checked_mul(4)
                    .and_then(|scale| n.checked_add(scale))
            })
            .and_then(|n| n.checked_add(input_bytes))
            .and_then(|n| n.checked_add(intermediate_bytes))
            .and_then(|n| {
                contribution_len
                    .checked_mul(4)
                    .and_then(|bytes| n.checked_add(bytes))
            })
            .ok_or("split grouped workspace byte overflow")?;
        let routes = GroupedRoutes::new_partition(s, global, 0, global, slots)?;
        let bytes = routes
            .bytes
            .checked_add(split_extra_bytes as u64)
            .ok_or("split grouped workspace byte overflow")?;
        Ok(Self {
            routes,
            input: HalfMirror::new(s, slots, hidden)?,
            intermediate: HalfMirror::new(s, slots, local_inter)?,
            contribution: s
                .alloc_zeros::<f32>(contribution_len)
                .map_err(|e| format!("split grouped contribution allocation: {e}"))?,
            bytes,
            split_scratch: Some(SplitScratch {
                h: s.alloc_zeros::<f32>(split_h_len)
                    .map_err(|e| format!("split grouped h allocation: {e}"))?,
                hq: s
                    .alloc_zeros::<u8>(split_h_len)
                    .map_err(|e| format!("split grouped hq allocation: {e}"))?,
                hs: s
                    .alloc_zeros::<f32>(split_hs_len)
                    .map_err(|e| format!("split grouped hs allocation: {e}"))?,
            }),
            plain_single: false,
            gu_fuse: false,
            phase: MatrixPhase::Idle,
            splitk_scratch: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn splitk(
        &mut self,
        gpu: &Gpu,
        table: &CudaSlice<u64>,
        output: u64,
        limit: f32,
        gu: bool,
        component: bool,
    ) -> Res<()> {
        let s = gpu.stream();
        let (input, out_f) = if gu {
            (&self.input, self.intermediate.cols)
        } else {
            (&self.intermediate, self.input.cols)
        };
        let live = self.routes.live_slots;
        if table.len() != self.routes.experts * 6
            || crate::moe_f16g_mode() < 2
            || crate::moe_f16g_sk_params().0 < 0
            || !crate::moe_f16g_direct_on(crate::QT_NVFP4_MODELOPT)
        {
            return Err(
                "split-K requires a complete local ModelOpt table and direct grouped visitor"
                    .into(),
            );
        }
        let needed = live
            .checked_mul(out_f)
            .and_then(|v| v.checked_mul(if gu { 32 } else { 16 }))
            .ok_or("split-K scratch overflow")?;
        if !component
            && self
                .splitk_scratch
                .as_ref()
                .is_none_or(|p| p.len() < needed)
        {
            let old_bytes = self.splitk_scratch.as_ref().map_or(0, |p| p.len() * 4);
            self.splitk_scratch = Some(
                s.alloc_zeros::<f32>(needed)
                    .map_err(|e| format!("split-K scratch: {e}"))?,
            );
            self.bytes = self.bytes - old_bytes as u64 + (needed * 4) as u64;
        }
        let partial = self
            .splitk_scratch
            .as_mut()
            .map_or(std::ptr::null_mut(), |p| p.device_ptr_mut(&s).0 as *mut f32);
        let launch = if component {
            crate::mmq_ffi::memra_moe_m1_splitk_component
        } else {
            crate::mmq_ffi::memra_moe_m1_splitk
        };
        let rc = unsafe {
            launch(
                table.device_ptr(&s).0 as *const u64,
                self.routes.experts as i32,
                self.routes.ids.device_ptr(&s).0 as *const i32,
                input.half.device_ptr(&s).0 as *const std::ffi::c_void,
                output as *mut f32,
                input.scale.device_ptr(&s).0 as *const f32,
                self.routes.macro1.device_ptr(&s).0 as *const f32,
                self.routes.macro3.device_ptr(&s).0 as *const f32,
                self.routes.weights.device_ptr(&s).0 as *const f32,
                self.routes.offsets.device_ptr(&s).0 as *const i32,
                self.routes.experts as i32,
                input.cols as i32,
                out_f as i32,
                limit,
                live as i32,
                gu as i32,
                partial,
                s.cu_stream().cast(),
            )
        };
        if rc != 0 {
            return Err(format!(
                "moe_m1_splitk gu={gu} component={component} rc={rc}"
            ));
        }
        if !component {
            let counter = if gu {
                &crate::MOE_M1_SPLITK_GU_DISPATCHES
            } else {
                &crate::MOE_M1_SPLITK_DOWN_DISPATCHES
            };
            counter.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    /// Source FP8 codes/scales are already produced on the token's owner.
    /// Preparation and its checks finish before either rank queues gate/up.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self,
        gpu: &Gpu,
        source: &EpCompute<'_>,
        scale2: &CudaSlice<f32>,
        scale2_host: &[f32],
        rows: usize,
        topk: usize,
        device_routes: bool,
    ) -> Res<bool> {
        if !matches!(self.phase, MatrixPhase::Idle | MatrixPhase::Failed) {
            return Err("matrix preparation would overwrite an unfinished chain".into());
        }
        self.phase = MatrixPhase::Failed;
        let slots = rows.checked_mul(topk).ok_or("matrix slot count overflow")?;
        self.plain_single = rows == 1;
        let hidden = self.input.cols;
        let inter = self.intermediate.cols;
        if rows == 0
            || rows > 512
            || slots > self.routes.capacity
            || source.xq.len() < rows * hidden
            || source.xs.len() < rows * hidden / 128
            || source.g1.len() < slots * inter
            || source.g3.len() < slots * inter
            || source.h.len() < slots * inter
            || source.hq.len() < slots * inter
            || source.hs.len() < slots * inter / 128
            || source.contribution.len() < slots * hidden
        {
            return Err("matrix chain input/workspace shape mismatch".into());
        }
        bind_matrix(gpu)?;
        let s = gpu.stream();
        let used_device = self.routes.prepare(
            &s,
            source.ids,
            source.weights,
            scale2,
            scale2_host,
            slots,
            topk,
            device_routes,
        )?;
        if self.routes.live_slots > 0 {
            self.input.gather(
                &s,
                source.xq,
                source.xs,
                Some(&self.routes.tokens),
                self.routes.live_slots,
            )?;
        }
        self.phase = MatrixPhase::Prepared;
        Ok(used_device)
    }

    pub fn set_gu_fuse_for_plain(&mut self, enabled: bool) {
        self.gu_fuse = enabled && self.plain_single;
    }

    fn output_ptrs(
        &mut self,
        out: &mut EpCompute<'_>,
        stream: &Arc<CudaStream>,
    ) -> (*mut f32, *mut u8, *mut f32) {
        if let Some(split) = self.split_scratch.as_mut() {
            (
                split.h.device_ptr_mut(stream).0 as *mut f32,
                split.hq.device_ptr_mut(stream).0 as *mut u8,
                split.hs.device_ptr_mut(stream).0 as *mut f32,
            )
        } else {
            (
                out.h.device_ptr_mut(stream).0 as *mut f32,
                out.hq.device_ptr_mut(stream).0 as *mut u8,
                out.hs.device_ptr_mut(stream).0 as *mut f32,
            )
        }
    }

    fn gather_intermediate(
        &mut self,
        stream: &Arc<CudaStream>,
        out: &EpCompute<'_>,
        live: usize,
    ) -> Res<()> {
        if let Some(split) = self.split_scratch.as_mut() {
            self.intermediate
                .gather(stream, &split.hq, &split.hs, None, live)
        } else {
            self.intermediate.gather(stream, out.hq, out.hs, None, live)
        }
    }

    /// No host readback in this stage: both ranks can queue gate/up before down.
    pub fn gate_up(
        &mut self,
        gpu: &Gpu,
        table: &CudaSlice<u64>,
        out: &mut EpCompute<'_>,
        limit: f32,
    ) -> Res<()> {
        if self.phase != MatrixPhase::Prepared {
            return Err("matrix gate/up requires prepared input".into());
        }
        self.phase = MatrixPhase::Failed;
        bind_matrix(gpu)?;
        let s = gpu.stream();
        let live = self.routes.live_slots;
        let (h_ptr, hq_ptr, hs_ptr) = self.output_ptrs(out, &s);
        if !route_validation_enabled() {
            s.memset_zeros(&mut self.contribution)
                .map_err(|e| format!("grouped contribution clear: {e}"))?;
        }
        if live > 0 {
            let fuse_gu = self.gu_fuse && crate::moe_f16g_gu_fuse_on() && crate::moe_f16g_tail_on();
            let gu_kind = select_gu_launch(
                fuse_gu,
                crate::moe_f16g_gu_m1_tc_on(),
                crate::moe_f16g_gu_half2_on(),
            );
            if self.plain_single && splitk_component_claim(gpu, true) {
                self.splitk(gpu, table, h_ptr as u64, limit, true, true)?;
            }
            if self.plain_single && crate::moe_m1_splitk_on() {
                if !fuse_gu {
                    return Err("split-K requires plain fused GU".into());
                }
                self.splitk(gpu, table, h_ptr as u64, limit, true, false)?;
            } else if let Some(gu_kind) = gu_kind {
                let rc = unsafe {
                    let launch = match gu_kind {
                        GuLaunchKind::M1Half2 => memra_moe_kq_gemm_sk_gu_m1_half2,
                        GuLaunchKind::Half2 => memra_moe_kq_gemm_sk_gu_half2,
                        GuLaunchKind::M1 => memra_moe_kq_gemm_sk_gu_m1,
                        GuLaunchKind::Scalar => memra_moe_kq_gemm_sk_gu,
                    };
                    launch(
                        table.device_ptr(&s).0 as *const u64,
                        self.routes.experts as i32,
                        self.routes.ids.device_ptr(&s).0 as *const i32,
                        self.input.half.device_ptr(&s).0 as *const std::ffi::c_void,
                        h_ptr,
                        self.input.scale.device_ptr(&s).0 as *const f32,
                        self.routes.macro1.device_ptr(&s).0 as *const f32,
                        self.routes.macro3.device_ptr(&s).0 as *const f32,
                        self.routes.weights.device_ptr(&s).0 as *const f32,
                        self.routes.offsets.device_ptr(&s).0 as *const i32,
                        self.routes.experts as i32,
                        self.input.cols as i32,
                        self.intermediate.cols as i32,
                        limit,
                        (self.input.cols / 2) as i64,
                        s.cu_stream().cast(),
                    )
                };
                if rc != 0 {
                    return Err(format!(
                        "{} rc={rc}",
                        match gu_kind {
                            GuLaunchKind::M1Half2 => "memra_moe_kq_gemm_sk_gu_m1_half2",
                            GuLaunchKind::Half2 => "memra_moe_kq_gemm_sk_gu_half2",
                            GuLaunchKind::M1 => "memra_moe_kq_gemm_sk_gu_m1",
                            GuLaunchKind::Scalar => "memra_moe_kq_gemm_sk_gu",
                        }
                    ));
                }
                if gu_receipt_features(gu_kind).0 {
                    // The combined M1+half2 enqueue is one CUDA launch, but it
                    // legitimately advances both existing feature receipts.
                    crate::MOE_F16G_GU_M1_TC_DISPATCHES
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                static SAID: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!(
                        "[dsv4-prefill-f16g] GU_FUSE ENGAGED: rows=1 live_slots={live} hidden={} inter={} ModelOpt NVFP4 f16-MMA class",
                        self.input.cols, self.intermediate.cols
                    );
                }
            } else {
                self.routes
                    .project(&s, table, 0, &self.input, self.intermediate.cols, out.g1)?;
                self.routes
                    .project(&s, table, 2, &self.input, self.intermediate.cols, out.g3)?;
                unsafe {
                    for (dst, scales) in [
                        (&mut *out.g1, &self.routes.macro1),
                        (&mut *out.g3, &self.routes.macro3),
                    ] {
                        dsv4_ffi::ck(
                            "matrix gate/up scale",
                            dsv4_ffi::memra_dsv4_scale_rows(
                                dst.device_ptr_mut(&s).0 as *mut f32,
                                scales.device_ptr(&s).0 as *const f32,
                                live as i32,
                                self.intermediate.cols as i32,
                                s.cu_stream().cast(),
                            ),
                        )?;
                    }
                    dsv4_ffi::ck(
                        "matrix weighted SwiGLU",
                        dsv4_ffi::memra_dsv4_swiglu(
                            out.g1.device_ptr(&s).0 as *const f32,
                            out.g3.device_ptr(&s).0 as *const f32,
                            h_ptr,
                            live as i32,
                            self.intermediate.cols as i32,
                            limit,
                            self.routes.weights.device_ptr(&s).0 as *const f32,
                            s.cu_stream().cast(),
                        ),
                    )?;
                }
            }
            unsafe {
                dsv4_ffi::ck(
                    "matrix intermediate FP8",
                    dsv4_ffi::memra_dsv4_act_quant_fp8(
                        h_ptr as *const f32,
                        hq_ptr as *mut std::ffi::c_void,
                        hs_ptr,
                        live as i32,
                        self.intermediate.cols as i32,
                        s.cu_stream().cast(),
                    ),
                )?;
            }
        }
        self.phase = MatrixPhase::UpQueued;
        Ok(())
    }

    pub fn down(&mut self, gpu: &Gpu, table: &CudaSlice<u64>, out: &mut EpCompute<'_>) -> Res<()> {
        if self.phase != MatrixPhase::UpQueued {
            return Err("matrix down requires queued gate/up".into());
        }
        self.phase = MatrixPhase::Failed;
        bind_matrix(gpu)?;
        let s = gpu.stream();
        let live = self.routes.live_slots;
        if live > 0 {
            self.gather_intermediate(&s, out, live)?;
            let component = self.plain_single && splitk_component_claim(gpu, false);
            let splitk = self.plain_single && crate::moe_m1_splitk_on();
            let output = self.contribution.device_ptr_mut(&s).0;
            if component {
                self.splitk(gpu, table, output, 0.0, false, true)?;
            }
            if splitk {
                self.splitk(gpu, table, output, 0.0, false, false)?;
            } else if self.plain_single
                && crate::moe_f16g_tail_on()
                && (crate::moe_f16g_m1_tc_on() || crate::moe_f16g_down_m1_half2_on())
            {
                self.routes.project_m1(
                    &s,
                    table,
                    &self.intermediate,
                    &mut self.contribution,
                    self.input.cols,
                    crate::moe_f16g_down_m1_half2_on(),
                )?;
            } else {
                self.routes.project(
                    &s,
                    table,
                    1,
                    &self.intermediate,
                    self.input.cols,
                    &mut self.contribution,
                )?;
            }
            unsafe {
                dsv4_ffi::ck(
                    "matrix down scale",
                    dsv4_ffi::memra_dsv4_scale_rows(
                        self.contribution.device_ptr_mut(&s).0 as *mut f32,
                        self.routes.macro2.device_ptr(&s).0 as *const f32,
                        live as i32,
                        self.input.cols as i32,
                        s.cu_stream().cast(),
                    ),
                )?;
                dsv4_ffi::ck(
                    "matrix original-slot scatter",
                    dsv4_ffi::memra_dsv4_scatter_rows(
                        self.contribution.device_ptr(&s).0 as *const f32,
                        out.contribution.device_ptr_mut(&s).0 as *mut f32,
                        self.routes.pairs.device_ptr(&s).0 as *const i32,
                        live as i32,
                        self.input.cols as i32,
                        s.cu_stream().cast(),
                    ),
                )?;
            }
        }
        self.phase = MatrixPhase::Idle;
        Ok(())
    }
}

pub(crate) struct GroupedRoutes {
    pub ids: CudaSlice<i32>,
    pub offsets: CudaSlice<i32>,
    pub pairs: CudaSlice<i32>,
    pub tokens: CudaSlice<i32>,
    pub weights: CudaSlice<f32>,
    pub macro1: CudaSlice<f32>,
    pub macro2: CudaSlice<f32>,
    pub macro3: CudaSlice<f32>,
    counts: CudaSlice<i32>,
    status: CudaSlice<i32>,
    pub host_offsets: Option<Vec<i32>>,
    pub max_m: i32,
    pub live_slots: usize,
    /// True only when a device/host route-count readback established the exact
    /// local live prefix. Validation-off device routing keeps `live_slots` as
    /// a launch upper bound, never as an observed count.
    pub live_slots_observed: bool,
    pub bytes: u64,
    experts: usize,
    global_experts: usize,
    first: usize,
    capacity: usize,
}

fn partition_shape(global: usize, first: usize, count: usize, slots: usize) -> Res<()> {
    if global == 0
        || global > 512
        || first >= global
        || count == 0
        || count > global - first
        || slots == 0
        || slots > i32::MAX as usize
    {
        return Err("invalid grouped route partition dimensions".into());
    }
    Ok(())
}

impl GroupedRoutes {
    pub fn matches_partition(&self, global: usize, first: usize, count: usize) -> bool {
        self.global_experts == global && self.first == first && self.experts == count
    }
    fn project(
        &self,
        s: &Arc<CudaStream>,
        table: &CudaSlice<u64>,
        projection: i32,
        input: &HalfMirror,
        out_f: usize,
        output: &mut CudaSlice<f32>,
    ) -> Res<()> {
        if table.len() != self.experts * 6
            || output.len() < self.live_slots * out_f
            || crate::moe_f16g_mode() < 2
            || crate::moe_f16g_sk_params().0 < 0
            || !crate::moe_f16g_direct_on(crate::QT_NVFP4_MODELOPT)
        {
            return Err(
                "matrix projection requires a complete local table and direct grouped visitor"
                    .into(),
            );
        }
        let (_, cross) = crate::moe_f16g_sk_params();
        let rc = unsafe {
            memra_moe_kq_gemm_sk(
                table.device_ptr(s).0 as *const u64,
                projection,
                self.experts as i32,
                self.ids.device_ptr(s).0 as *const i32,
                input.half.device_ptr(s).0 as *const std::ffi::c_void,
                output.device_ptr_mut(s).0 as *mut f32,
                input.scale.device_ptr(s).0 as *const f32,
                self.offsets.device_ptr(s).0 as *const i32,
                self.host_offsets
                    .as_ref()
                    .map_or(std::ptr::null(), |o| o.as_ptr()),
                self.experts as i32,
                self.max_m,
                input.cols as i32,
                out_f as i32,
                crate::QT_NVFP4_MODELOPT,
                cross,
                crate::moe_f16g_tail_on() as i32,
                (input.cols / 2) as i64,
                s.cu_stream().cast(),
            )
        };
        if rc != 0 {
            return Err(format!("matrix projection {projection} rc={rc}"));
        }
        Ok(())
    }

    fn project_m1(
        &self,
        s: &Arc<CudaStream>,
        table: &CudaSlice<u64>,
        input: &HalfMirror,
        output: &mut CudaSlice<f32>,
        out_f: usize,
        half2: bool,
    ) -> Res<()> {
        if table.len() != self.experts * 6
            || output.len() < self.live_slots * out_f
            || crate::moe_f16g_mode() < 2
            || crate::moe_f16g_sk_params().0 < 0
            || !crate::moe_f16g_direct_on(crate::QT_NVFP4_MODELOPT)
        {
            return Err(
                "matrix m1-tc projection requires a complete local table and direct visitor".into(),
            );
        }
        let rc = unsafe {
            let launch = if half2 {
                memra_moe_kq_gemm_sk_m1_half2
            } else {
                memra_moe_kq_gemm_sk_m1
            };
            launch(
                table.device_ptr(s).0 as *const u64,
                self.experts as i32,
                self.ids.device_ptr(s).0 as *const i32,
                input.half.device_ptr(s).0 as *const std::ffi::c_void,
                output.device_ptr_mut(s).0 as *mut f32,
                input.scale.device_ptr(s).0 as *const f32,
                self.offsets.device_ptr(s).0 as *const i32,
                self.experts as i32,
                input.cols as i32,
                out_f as i32,
                (input.cols / 2) as i64,
                s.cu_stream().cast(),
            )
        };
        if rc != 0 {
            return Err(format!("matrix m1-tc projection rc={rc}"));
        }
        Ok(())
    }

    pub fn new_partition(
        s: &Arc<CudaStream>,
        global_experts: usize,
        first: usize,
        experts: usize,
        slots: usize,
    ) -> Res<Self> {
        partition_shape(global_experts, first, experts, slots)?;
        let i = |n| {
            s.alloc_zeros::<i32>(n)
                .map_err(|e| format!("grouped route i32: {e}"))
        };
        let f = |n| {
            s.alloc_zeros::<f32>(n)
                .map_err(|e| format!("grouped route f32: {e}"))
        };
        Ok(Self {
            ids: i(experts)?,
            offsets: i(experts + 1)?,
            pairs: i(slots)?,
            tokens: i(slots)?,
            weights: f(slots)?,
            macro1: f(slots)?,
            macro2: f(slots)?,
            macro3: f(slots)?,
            counts: i(experts)?,
            status: i(1)?,
            host_offsets: None,
            max_m: slots as i32,
            live_slots: 0,
            live_slots_observed: false,
            bytes: ((3 * experts + 2 + 6 * slots) * 4) as u64,
            experts,
            global_experts,
            first,
            capacity: slots,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self,
        s: &Arc<CudaStream>,
        selected: &CudaSlice<i32>,
        weights: &CudaSlice<f32>,
        scale2: &CudaSlice<f32>,
        scale2_host: &[f32],
        slots: usize,
        topk: usize,
        device: bool,
    ) -> Res<bool> {
        self.live_slots = 0;
        self.live_slots_observed = false;
        let partition = self.first != 0 || self.experts != self.global_experts;
        if slots == 0
            || slots > self.capacity
            || topk == 0
            || topk > self.global_experts
            || !slots.is_multiple_of(topk)
            || (partition && slots / topk > 512)
            || selected.len() < slots
            || weights.len() < slots
            || scale2.len() != self.global_experts * 3
            || scale2_host.len() != self.global_experts * 3
        {
            return Err("grouped route input/workspace shape mismatch".into());
        }
        self.host_offsets = None;
        self.max_m = slots as i32; // Device arm uses this only as a nonzero upper bound.
        if device {
            let rc = if partition {
                unsafe {
                    dsv4_ffi::memra_dsv4_grouped_routes_partition(
                        selected.device_ptr(s).0 as *const i32,
                        weights.device_ptr(s).0 as *const f32,
                        scale2.device_ptr(s).0 as *const f32,
                        self.counts.device_ptr_mut(s).0 as *mut i32,
                        self.offsets.device_ptr_mut(s).0 as *mut i32,
                        self.ids.device_ptr_mut(s).0 as *mut i32,
                        self.pairs.device_ptr_mut(s).0 as *mut i32,
                        self.tokens.device_ptr_mut(s).0 as *mut i32,
                        self.weights.device_ptr_mut(s).0 as *mut f32,
                        self.macro1.device_ptr_mut(s).0 as *mut f32,
                        self.macro2.device_ptr_mut(s).0 as *mut f32,
                        self.macro3.device_ptr_mut(s).0 as *mut f32,
                        self.status.device_ptr_mut(s).0 as *mut i32,
                        slots as i32,
                        self.global_experts as i32,
                        self.first as i32,
                        self.experts as i32,
                        topk as i32,
                        s.cu_stream() as *mut std::ffi::c_void,
                    )
                }
            } else {
                unsafe {
                    dsv4_ffi::memra_dsv4_grouped_routes(
                        selected.device_ptr(s).0 as *const i32,
                        weights.device_ptr(s).0 as *const f32,
                        scale2.device_ptr(s).0 as *const f32,
                        self.counts.device_ptr_mut(s).0 as *mut i32,
                        self.offsets.device_ptr_mut(s).0 as *mut i32,
                        self.ids.device_ptr_mut(s).0 as *mut i32,
                        self.pairs.device_ptr_mut(s).0 as *mut i32,
                        self.tokens.device_ptr_mut(s).0 as *mut i32,
                        self.weights.device_ptr_mut(s).0 as *mut f32,
                        self.macro1.device_ptr_mut(s).0 as *mut f32,
                        self.macro2.device_ptr_mut(s).0 as *mut f32,
                        self.macro3.device_ptr_mut(s).0 as *mut f32,
                        self.status.device_ptr_mut(s).0 as *mut i32,
                        slots as i32,
                        self.experts as i32,
                        topk as i32,
                        s.cu_stream() as *mut std::ffi::c_void,
                    )
                }
            };
            if rc != 0 {
                return Err(format!("grouped route kernel rc={rc}"));
            }
            if !route_validation_enabled() {
                // The device prefix remains authoritative for the visitor. The
                // compacted slot arrays are cleared by the route count kernel;
                // gather/scatter treat their -1 tail as inert. This gate removes
                // only the host status/live-count readback and synchronize. The
                // full slot count is a launch upper bound, not an observed local
                // live count; callers must not use it for occupancy statistics.
                self.live_slots = slots;
                self.live_slots_observed = false;
                return Ok(true);
            }
            // Initial admission retains a scalar fail-closed check. This is NOT
            // yet a fully graphable MoE; the FP8/half checks also still synchronize.
            let mut status = [0i32];
            s.memcpy_dtoh(&self.status, &mut status[..])
                .map_err(|e| format!("route status: {e}"))?;
            let mut live = [slots as i32];
            if partition {
                s.memcpy_dtoh(
                    &self.offsets.slice(self.experts..self.experts + 1),
                    &mut live[..],
                )
                .map_err(|e| format!("route live count: {e}"))?;
            }
            s.synchronize()
                .map_err(|e| format!("route status sync: {e}"))?;
            if status[0] != 0 {
                return Err("grouped route contains an invalid expert id".into());
            }
            if live[0] < 0 || live[0] as usize > slots {
                return Err("grouped route live count outside input slots".into());
            }
            self.live_slots = live[0] as usize;
            self.live_slots_observed = true;
            return Ok(true);
        }

        let mut sel = vec![0i32; slots];
        let mut route_weights = vec![0f32; slots];
        s.memcpy_dtoh(&selected.slice(..slots), &mut sel)
            .map_err(|e| format!("route ids: {e}"))?;
        s.memcpy_dtoh(&weights.slice(..slots), &mut route_weights)
            .map_err(|e| format!("route weights: {e}"))?;
        s.synchronize()
            .map_err(|e| format!("host routing sync: {e}"))?;
        let mut groups = vec![Vec::new(); self.experts];
        for (p, &expert) in sel.iter().enumerate() {
            let expert = usize::try_from(expert).map_err(|_| "negative expert id")?;
            if expert >= self.global_experts {
                return Err("expert id outside global route table".into());
            }
            if (self.first..self.first + self.experts).contains(&expert) {
                groups[expert - self.first].push(p as i32);
            }
        }
        let mut offsets = vec![0i32];
        let mut pairs = Vec::with_capacity(slots);
        self.max_m = 0;
        for group in groups {
            self.max_m = self.max_m.max(group.len() as i32);
            pairs.extend(group);
            offsets.push(pairs.len() as i32);
        }
        let tokens: Vec<i32> = pairs.iter().map(|p| p / topk as i32).collect();
        let reordered: Vec<f32> = pairs.iter().map(|&p| route_weights[p as usize]).collect();
        let macro_for = |proj| {
            pairs
                .iter()
                .map(|&p| scale2_host[sel[p as usize] as usize * 3 + proj])
                .collect::<Vec<f32>>()
        };
        let ids: Vec<i32> = (0..self.experts as i32).collect();
        s.memcpy_htod(&ids, &mut self.ids)
            .map_err(|e| format!("route ids upload: {e}"))?;
        s.memcpy_htod(&offsets, &mut self.offsets)
            .map_err(|e| format!("route offsets upload: {e}"))?;
        let live = pairs.len();
        if live > 0 {
            s.memcpy_htod(&pairs, &mut self.pairs.slice_mut(..live))
                .map_err(|e| format!("route pairs upload: {e}"))?;
            s.memcpy_htod(&tokens, &mut self.tokens.slice_mut(..live))
                .map_err(|e| format!("route tokens upload: {e}"))?;
            s.memcpy_htod(&reordered, &mut self.weights.slice_mut(..live))
                .map_err(|e| format!("route weights upload: {e}"))?;
            for (dst, proj) in [
                (&mut self.macro1, 0),
                (&mut self.macro2, 1),
                (&mut self.macro3, 2),
            ] {
                s.memcpy_htod(&macro_for(proj), &mut dst.slice_mut(..live))
                    .map_err(|e| format!("route scales upload: {e}"))?;
            }
        }
        self.host_offsets = Some(offsets);
        self.live_slots = live;
        self.live_slots_observed = true;
        Ok(false)
    }
}

#[cfg(test)]
mod gu_dispatch_selection_tests {
    use super::{GuLaunchKind, gu_receipt_features, select_gu_launch};

    #[test]
    fn existing_doors_compose_only_on_plain_single_path() {
        assert_eq!(select_gu_launch(false, false, false), None);
        assert_eq!(select_gu_launch(false, true, true), None);
        assert_eq!(
            select_gu_launch(true, false, false),
            Some(GuLaunchKind::Scalar)
        );
        assert_eq!(select_gu_launch(true, true, false), Some(GuLaunchKind::M1));
        assert_eq!(
            select_gu_launch(true, false, true),
            Some(GuLaunchKind::Half2)
        );
        assert_eq!(
            select_gu_launch(true, true, true),
            Some(GuLaunchKind::M1Half2)
        );
    }

    #[test]
    fn combined_kind_is_distinct_from_each_single_feature() {
        assert_ne!(
            select_gu_launch(true, true, true),
            select_gu_launch(true, true, false)
        );
        assert_ne!(
            select_gu_launch(true, true, true),
            select_gu_launch(true, false, true)
        );
        assert_eq!(gu_receipt_features(GuLaunchKind::M1Half2), (true, true));
    }
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "requires one CUDA GPU; GU-M1 component identity only"]
    fn cuda_gu_m1_matches_gu_reference() {
        use super::{GroupedWork, modelopt_table};
        use crate::dsv4_ep::{EpCompute, EpScratch};
        use crate::dsv4_ffi as k;
        use cudarc::driver::{DevicePtr, DevicePtrMut};

        fn view(x: &mut EpScratch) -> EpCompute<'_> {
            EpCompute {
                xq: &x.xq,
                xs: &x.xs,
                ids: &x.ids,
                weights: &x.weights,
                g1: &mut x.g1,
                g3: &mut x.g3,
                h: &mut x.h,
                hq: &mut x.hq,
                hs: &mut x.hs,
                contribution: &mut x.contribution,
            }
        }
        fn bits(values: &[f32]) -> Vec<u32> {
            values.iter().map(|value| value.to_bits()).collect()
        }

        assert!(crate::moe_f16g_mode() >= 2);
        assert!(crate::moe_f16g_direct_on(crate::QT_NVFP4_MODELOPT));
        let gpu = memra_runtime::Gpu::new(0).unwrap();
        let s = gpu.stream();
        let (ne, hidden, inter, topk) = (16, 4096, 2048, 6);
        let slots = topk;
        let wb = hidden * inter / 2;
        let sb = hidden * inter / 16;
        let weight_data: Vec<u8> = (0..ne * 3 * wb)
            .map(|i| ((i * 37 + 17 + (i / wb) * 13) % 256) as u8)
            .collect();
        let scale_data: Vec<u8> = (0..ne * 3 * sb)
            .map(|i| ((5 + (i % 4)) << 3) as u8)
            .collect();
        let weights = s.clone_htod(&weight_data).unwrap();
        let scales = s.clone_htod(&scale_data).unwrap();
        let table = modelopt_table(&s, &weights, &scales, ne, hidden, inter).unwrap();
        let scale2_host = vec![1.0f32; ne * 3];
        let scale2 = s.clone_htod(&scale2_host).unwrap();
        let input: Vec<f32> = (0..hidden)
            .map(|i| ((i % 31) as f32 - 15.0) / 16.0)
            .collect();
        let x = s.clone_htod(&input).unwrap();
        let selected: Vec<i32> = (0..slots).map(|i| i as i32).collect();
        let route_weights: Vec<f32> = (0..slots).map(|i| (i + 1) as f32 / 32.0).collect();
        let mut reference = EpScratch::new(&gpu, &gpu, 1, topk, hidden, inter, None).unwrap();
        let mut candidate = EpScratch::new(&gpu, &gpu, 1, topk, hidden, inter, None).unwrap();
        for scratch in [&mut reference, &mut candidate] {
            s.memcpy_htod(&selected, &mut scratch.ids.slice_mut(..slots))
                .unwrap();
            s.memcpy_htod(&route_weights, &mut scratch.weights.slice_mut(..slots))
                .unwrap();
            unsafe {
                k::ck(
                    "GU-M1 fixture FP8 input",
                    k::memra_dsv4_act_quant_fp8(
                        x.device_ptr(&s).0 as *const f32,
                        scratch.xq.device_ptr_mut(&s).0 as *mut std::ffi::c_void,
                        scratch.xs.device_ptr_mut(&s).0 as *mut f32,
                        1,
                        hidden as i32,
                        s.cu_stream().cast(),
                    ),
                )
                .unwrap();
            }
        }
        let mut reference_work = GroupedWork::new(&s, ne, slots, hidden, inter).unwrap();
        let mut candidate_work = GroupedWork::new(&s, ne, slots, hidden, inter).unwrap();
        crate::set_moe_f16g_gu_fuse_for_gate(true);
        crate::set_moe_f16g_gu_m1_tc_for_gate(false);
        reference_work
            .prepare(
                &gpu,
                &view(&mut reference),
                &scale2,
                &scale2_host,
                1,
                topk,
                true,
            )
            .unwrap();
        reference_work.set_gu_fuse_for_plain(true);
        reference_work
            .gate_up(&gpu, &table, &mut view(&mut reference), 6.0)
            .unwrap();
        let reference_h = s.clone_dtoh(&reference.h.slice(..slots * inter)).unwrap();
        reference_work
            .down(&gpu, &table, &mut view(&mut reference))
            .unwrap();
        let reference_contribution = s
            .clone_dtoh(&reference.contribution.slice(..slots * hidden))
            .unwrap();

        let before = crate::moe_f16g_gu_m1_tc_dispatches();
        crate::set_moe_f16g_gu_m1_tc_for_gate(true);
        candidate_work
            .prepare(
                &gpu,
                &view(&mut candidate),
                &scale2,
                &scale2_host,
                1,
                topk,
                true,
            )
            .unwrap();
        candidate_work.set_gu_fuse_for_plain(true);
        candidate_work
            .gate_up(&gpu, &table, &mut view(&mut candidate), 6.0)
            .unwrap();
        let candidate_h = s.clone_dtoh(&candidate.h.slice(..slots * inter)).unwrap();
        candidate_work
            .down(&gpu, &table, &mut view(&mut candidate))
            .unwrap();
        let candidate_contribution = s
            .clone_dtoh(&candidate.contribution.slice(..slots * hidden))
            .unwrap();
        let after = crate::moe_f16g_gu_m1_tc_dispatches();
        crate::clear_moe_f16g_gu_m1_tc_for_gate();
        crate::clear_moe_f16g_gu_fuse_for_gate();
        assert_eq!(after - before, 1, "GU-M1 launcher engagement");
        assert_eq!(bits(&reference_h), bits(&candidate_h), "GU H identity");
        assert_eq!(
            bits(&reference_contribution),
            bits(&candidate_contribution),
            "GU routed contribution identity"
        );
        println!(
            "PASS GU-M1 component identity: h_bits={} contribution_bits={} engagement_delta={}",
            reference_h.len(),
            reference_contribution.len(),
            after - before
        );
    }

    #[test]
    #[ignore = "requires one CUDA GPU; correctness only, run with matrix visitor enabled"]
    fn cuda_matrix_chain_full_bank_equals_two_partitions() {
        use super::{GroupedWork, modelopt_table};
        use crate::dsv4_ep::{EpCompute, EpScratch};
        use crate::dsv4_ffi as k;
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        fn view(x: &mut EpScratch) -> EpCompute<'_> {
            EpCompute {
                xq: &x.xq,
                xs: &x.xs,
                ids: &x.ids,
                weights: &x.weights,
                g1: &mut x.g1,
                g3: &mut x.g3,
                h: &mut x.h,
                hq: &mut x.hq,
                hs: &mut x.hs,
                contribution: &mut x.contribution,
            }
        }
        fn poison(gpu: &memra_runtime::Gpu, scratch: &mut EpScratch, work: &mut GroupedWork) {
            let s = gpu.stream();
            for dst in [
                &mut scratch.g1,
                &mut scratch.g3,
                &mut scratch.h,
                &mut scratch.contribution,
            ] {
                s.memcpy_htod(&vec![f32::NAN; dst.len()], dst).unwrap();
            }
            for dst in [&mut work.input.half, &mut work.intermediate.half] {
                let bytes: Vec<u8> = (0..dst.len())
                    .map(|i| if i % 2 == 0 { 0 } else { 0x7e })
                    .collect();
                s.memcpy_htod(&bytes, dst).unwrap();
            }
        }
        assert!(crate::moe_f16g_mode() >= 2 && crate::moe_f16g_direct_on(crate::QT_NVFP4_MODELOPT));
        let gpu = memra_runtime::Gpu::new(0).unwrap();
        let s = gpu.stream();
        let (ne, hidden, inter, topk, max_rows) = (16, 4096, 2048, 6, 32);
        let wb = hidden * inter / 2;
        let sb = hidden * inter / 16;
        let weight_data: Vec<u8> = (0..ne * 3 * wb)
            .map(|i| {
                let expert = i / (3 * wb);
                let projection = (i / wb) % 3;
                ((i * 37 + 17 + expert * 13 + projection * 59) % 256) as u8
            })
            .collect();
        let scale_data: Vec<u8> = (0..ne * 3 * sb)
            .map(|i| {
                let expert = i / (3 * sb);
                let projection = (i / sb) % 3;
                ((5 + (i + expert * 7 + projection * 3) % 4) << 3) as u8
            })
            .collect();
        let weights = s.clone_htod(&weight_data).unwrap();
        let scales = s.clone_htod(&scale_data).unwrap();
        drop(weight_data);
        drop(scale_data);
        let full_table = modelopt_table(&s, &weights, &scales, ne, hidden, inter).unwrap();
        let mut shard_w = Vec::new();
        let mut shard_s = Vec::new();
        let mut shard_tables = Vec::new();
        for first in [0, ne / 2] {
            let mut w = s.alloc_zeros::<u8>(ne / 2 * 3 * wb).unwrap();
            let mut sc = s.alloc_zeros::<u8>(ne / 2 * 3 * sb).unwrap();
            s.memcpy_dtod(
                &weights.slice(first * 3 * wb..(first + ne / 2) * 3 * wb),
                &mut w,
            )
            .unwrap();
            s.memcpy_dtod(
                &scales.slice(first * 3 * sb..(first + ne / 2) * 3 * sb),
                &mut sc,
            )
            .unwrap();
            shard_tables.push(modelopt_table(&s, &w, &sc, ne / 2, hidden, inter).unwrap());
            shard_w.push(w);
            shard_s.push(sc);
        }
        let scale2_host: Vec<f32> = (0..ne * 3)
            .map(|i| 2.0_f32.powi((i % 7) as i32 - 4))
            .collect();
        let scale2 = s.clone_htod(&scale2_host).unwrap();
        let input: Vec<f32> = (0..max_rows * hidden)
            .map(|i| ((i % 31) as f32 - 15.0) / 16.0 * 2.0_f32.powi(((i / 128) % 9) as i32 - 4))
            .collect();
        let x = s.clone_htod(&input).unwrap();
        let mut full = EpScratch::new(&gpu, &gpu, max_rows, topk, hidden, inter, None).unwrap();
        let mut a = EpScratch::new(&gpu, &gpu, max_rows, topk, hidden, inter, None).unwrap();
        let mut b = EpScratch::new(&gpu, &gpu, max_rows, topk, hidden, inter, None).unwrap();
        let mut full_work = GroupedWork::new(&s, ne, max_rows * topk, hidden, inter).unwrap();
        let mut a_work =
            GroupedWork::new_partition(&s, ne, 0, ne / 2, max_rows * topk, hidden, inter).unwrap();
        let mut b_work =
            GroupedWork::new_partition(&s, ne, ne / 2, ne / 2, max_rows * topk, hidden, inter)
                .unwrap();
        for rows in [32, 1, 6, 32] {
            let slots = rows * topk;
            for pattern in 0..3 {
                let selected: Vec<i32> = (0..slots)
                    .map(|p| match pattern {
                        0 => p % topk,
                        1 => ne / 2 + p % topk,
                        _ => (p * 5 + 1) % ne,
                    } as i32)
                    .collect();
                let routing: Vec<f32> = (0..slots)
                    .map(|p| {
                        if p % 5 == 0 {
                            -0.0
                        } else {
                            (p % 7 + 1) as f32 / 32.0
                        }
                    })
                    .collect();
                for scratch in [&mut full, &mut a, &mut b] {
                    s.memcpy_htod(&selected, &mut scratch.ids.slice_mut(..slots))
                        .unwrap();
                    s.memcpy_htod(&routing, &mut scratch.weights.slice_mut(..slots))
                        .unwrap();
                    unsafe {
                        k::ck(
                            "chain fixture FP8 input",
                            k::memra_dsv4_act_quant_fp8(
                                x.device_ptr(&s).0 as *const f32,
                                scratch.xq.device_ptr_mut(&s).0 as *mut std::ffi::c_void,
                                scratch.xs.device_ptr_mut(&s).0 as *mut f32,
                                rows as i32,
                                hidden as i32,
                                s.cu_stream().cast(),
                            ),
                        )
                        .unwrap();
                    }
                }
                poison(&gpu, &mut full, &mut full_work);
                poison(&gpu, &mut a, &mut a_work);
                poison(&gpu, &mut b, &mut b_work);
                full_work
                    .prepare(
                        &gpu,
                        &view(&mut full),
                        &scale2,
                        &scale2_host,
                        rows,
                        topk,
                        true,
                    )
                    .unwrap();
                full_work
                    .gate_up(&gpu, &full_table, &mut view(&mut full), 6.0)
                    .unwrap();
                full_work
                    .down(&gpu, &full_table, &mut view(&mut full))
                    .unwrap();
                a_work
                    .prepare(&gpu, &view(&mut a), &scale2, &scale2_host, rows, topk, true)
                    .unwrap();
                b_work
                    .prepare(&gpu, &view(&mut b), &scale2, &scale2_host, rows, topk, true)
                    .unwrap();
                a_work
                    .gate_up(&gpu, &shard_tables[0], &mut view(&mut a), 6.0)
                    .unwrap();
                b_work
                    .gate_up(&gpu, &shard_tables[1], &mut view(&mut b), 6.0)
                    .unwrap();
                a_work
                    .down(&gpu, &shard_tables[0], &mut view(&mut a))
                    .unwrap();
                b_work
                    .down(&gpu, &shard_tables[1], &mut view(&mut b))
                    .unwrap();
                assert_eq!(a_work.routes.live_slots + b_work.routes.live_slots, slots);
                unsafe {
                    k::ck(
                        "chain fixture original-slot merge",
                        k::memra_dsv4_ep_merge_slots(
                            a.contribution.device_ptr_mut(&s).0 as *mut f32,
                            b.contribution.device_ptr(&s).0 as *const f32,
                            a.ids.device_ptr(&s).0 as *const i32,
                            slots as i32,
                            hidden as i32,
                            (ne / 2) as i32,
                            (ne / 2) as i32,
                            s.cu_stream().cast(),
                        ),
                    )
                    .unwrap();
                }
                let expected = s
                    .clone_dtoh(&full.contribution.slice(..slots * hidden))
                    .unwrap();
                let actual = s
                    .clone_dtoh(&a.contribution.slice(..slots * hidden))
                    .unwrap();
                assert!(expected.iter().chain(&actual).all(|x| x.is_finite()));
                assert_eq!(
                    expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    "complete weighted FP8-QAT matrix chain mismatch rows={rows} pattern={pattern}"
                );
                assert!(
                    a_work
                        .down(&gpu, &shard_tables[0], &mut view(&mut a))
                        .is_err(),
                    "double-down must refuse"
                );
                println!(
                    "EXACT matrix chain hidden={hidden} inter={inter} rows={rows} pattern={pattern} input/intermediate FP8, weighted SwiGLU, down, original slots and poisoned unused tails"
                );
            }
        }
        drop((shard_tables, shard_w, shard_s));
    }

    #[test]
    fn modelopt_table_addresses_expert_major_banks_by_projection() {
        let wb = 128 * 256 / 2;
        let sb = 128 * 256 / 16;
        let w = 0x1000;
        let s = 0x100000;
        let table = super::modelopt_pointers(w, s, 6 * wb, 6 * sb, 2, 128, 256).unwrap();
        assert_eq!(
            table,
            vec![
                w,
                w + (3 * wb) as u64,
                s,
                s + (3 * sb) as u64,
                w + wb as u64,
                w + (4 * wb) as u64,
                s + sb as u64,
                s + (4 * sb) as u64,
                w + (2 * wb) as u64,
                w + (5 * wb) as u64,
                s + (2 * sb) as u64,
                s + (5 * sb) as u64,
            ]
        );
        assert!(super::modelopt_pointers(w, s, 6 * wb - 1, 6 * sb, 2, 128, 256).is_err());
        assert!(super::modelopt_pointers(w, s, 6 * wb, 6 * sb + 1, 2, 128, 256).is_err());
        assert!(super::modelopt_pointers(u64::MAX, s, 6 * wb, 6 * sb, 2, 128, 256).is_err());
        assert!(super::modelopt_pointers(w, u64::MAX, 6 * wb, 6 * sb, 2, 128, 256).is_err());
        for (ne, hidden, inter) in [
            (0, 128, 256),
            (513, 128, 256),
            (2, 0, 256),
            (2, 127, 256),
            (2, 128, 255),
            (2, usize::MAX - 127, 256),
        ] {
            assert!(super::modelopt_pointers(w, s, 6 * wb, 6 * sb, ne, hidden, inter).is_err());
        }
    }

    use super::{
        GroupedWork, HalfMirror, ensure_program, mirror_bytes, partition_shape, resolve,
        resolve_program,
    };

    #[test]
    fn expert_partition_bounds_do_not_wrap() {
        for (global, first, count) in [(256, 0, 128), (256, 128, 128), (256, 255, 1), (1, 0, 1)] {
            assert_eq!(partition_shape(global, first, count, 6), Ok(()));
        }
        for (global, first, count, slots) in [
            (0, 0, 1, 6),
            (513, 0, 1, 6),
            (256, 256, 1, 6),
            (256, 255, 2, 6),
            (256, 0, 0, 6),
            (256, usize::MAX, 1, 6),
            (256, 0, usize::MAX, 6),
            (256, 0, 128, 0),
            (256, 0, 128, usize::MAX),
        ] {
            assert!(partition_shape(global, first, count, slots).is_err());
        }
    }

    #[test]
    #[ignore = "requires CUDA; run under the rig lock, correctness only"]
    fn cuda_partition_routes_match_host_and_reject_global_ids() {
        use cudarc::driver::CudaContext;
        let ctx = CudaContext::new(0).unwrap();
        let s = ctx.default_stream();
        for (global, first, count) in [
            (256, 0, 128),
            (256, 128, 128),
            (256, 255, 1),
            (8, 3, 2),
            (1, 0, 1),
        ] {
            let topk = global.min(6);
            let capacity = 32 * topk;
            let mut host =
                super::GroupedRoutes::new_partition(&s, global, first, count, capacity).unwrap();
            let mut device =
                super::GroupedRoutes::new_partition(&s, global, first, count, capacity).unwrap();
            let mut selected = s.alloc_zeros::<i32>(capacity).unwrap();
            let weight_values: Vec<f32> = (0..capacity)
                .map(|p| if p % 5 == 0 { -0.0 } else { p as f32 / 32.0 })
                .collect();
            let scale_values: Vec<f32> = (0..3 * global)
                .map(|p| 2.0_f32.powi((p % 9) as i32 - 8))
                .collect();
            let weights = s.clone_htod(&weight_values).unwrap();
            let scales = s.clone_htod(&scale_values).unwrap();
            for rows in [32, 1, 3, 32] {
                let slots = rows * topk;
                for pattern in 0..4 {
                    let outsider = if first > 0 {
                        first - 1
                    } else if count < global {
                        count
                    } else {
                        0
                    };
                    let ids: Vec<i32> = (0..slots)
                        .map(|p| match pattern {
                            0 => first,
                            1 => outsider,
                            2 => (p * 97 + 13) % global,
                            _ => first + count - 1,
                        } as i32)
                        .collect();
                    s.memcpy_htod(&ids, &mut selected.slice_mut(..slots))
                        .unwrap();
                    let live = ids
                        .iter()
                        .filter(|&&id| (first..first + count).contains(&(id as usize)))
                        .count();
                    assert!(
                        !host
                            .prepare(
                                &s,
                                &selected,
                                &weights,
                                &scales,
                                &scale_values,
                                slots,
                                topk,
                                false
                            )
                            .unwrap()
                    );
                    assert!(
                        device
                            .prepare(
                                &s,
                                &selected,
                                &weights,
                                &scales,
                                &scale_values,
                                slots,
                                topk,
                                true
                            )
                            .unwrap()
                    );
                    assert_eq!(host.live_slots, live);
                    assert_eq!(device.live_slots, live);
                    assert!(host.host_offsets.is_some() && device.host_offsets.is_none());
                    for (a, b) in [(&host.ids, &device.ids), (&host.offsets, &device.offsets)] {
                        assert_eq!(s.clone_dtoh(a).unwrap(), s.clone_dtoh(b).unwrap());
                    }
                    if live > 0 {
                        for (a, b) in [(&host.pairs, &device.pairs), (&host.tokens, &device.tokens)]
                        {
                            assert_eq!(
                                s.clone_dtoh(&a.slice(..live)).unwrap(),
                                s.clone_dtoh(&b.slice(..live)).unwrap()
                            );
                        }
                        for (a, b) in [
                            (&host.weights, &device.weights),
                            (&host.macro1, &device.macro1),
                            (&host.macro2, &device.macro2),
                            (&host.macro3, &device.macro3),
                        ] {
                            let bits =
                                |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
                            assert_eq!(
                                bits(s.clone_dtoh(&a.slice(..live)).unwrap()),
                                bits(s.clone_dtoh(&b.slice(..live)).unwrap())
                            );
                        }
                    }
                }
            }
            for invalid in [-1, global as i32, i32::MAX] {
                let mut ids = vec![first as i32; capacity];
                ids[capacity - 1] = invalid;
                s.memcpy_htod(&ids, &mut selected).unwrap();
                for (routes, on_device) in [(&mut host, false), (&mut device, true)] {
                    assert!(
                        routes
                            .prepare(
                                &s,
                                &selected,
                                &weights,
                                &scales,
                                &scale_values,
                                capacity,
                                topk,
                                on_device
                            )
                            .is_err()
                    );
                    assert_eq!(
                        routes.live_slots, 0,
                        "failed preparation must not publish a live prefix"
                    );
                    // The stale invalid id is outside this smaller live input.
                    routes
                        .prepare(
                            &s,
                            &selected,
                            &weights,
                            &scales,
                            &scale_values,
                            topk,
                            topk,
                            on_device,
                        )
                        .unwrap();
                    assert_eq!(routes.live_slots, topk);
                }
            }
        }
        println!(
            "PASS partitioned Rust routes: host/device bits, global-id guards, empty ranks and live-prefix shrink/grow"
        );
    }

    #[test]
    fn matrix_mirror_byte_accounting_is_bounded() {
        assert_eq!(mirror_bytes(6, 4096), Ok(6 * (4096 * 2 + 8)));
        assert_eq!(mirror_bytes(3072, 2048), Ok(3072 * (2048 * 2 + 8)));
        for (rows, cols) in [
            (0, 128),
            (1, 0),
            (1, 127),
            (1, 129),
            (usize::MAX, 128),
            (1, usize::MAX),
        ] {
            assert!(mirror_bytes(rows, cols).is_err());
        }
    }

    #[test]
    #[ignore = "requires CUDA; run under the rig lock, correctness only"]
    fn cuda_matrix_mirrors_reuse_storage_and_clear_live_status() {
        use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
        let ctx = CudaContext::new(0).expect("context");
        // Same ownership discipline as the DSV4 loader, before any allocation.
        unsafe { ctx.disable_event_tracking() };
        let s = ctx.default_stream();
        let cols = 4096;
        let mut codes_host: Vec<u8> = (0..3 * cols).map(|i| ((i * 53 + 7) % 126) as u8).collect();
        let scales_host: Vec<f32> = (0..3 * cols / 128)
            .map(|i| 2.0_f32.powi((i % 11) as i32 - 5))
            .collect();
        let mut codes = s.alloc_zeros::<u8>(codes_host.len()).unwrap();
        let mut scales = s.alloc_zeros::<f32>(scales_host.len()).unwrap();
        let mut ids = s.alloc_zeros::<i32>(8).unwrap();
        s.memcpy_htod(&codes_host, &mut codes).unwrap();
        s.memcpy_htod(&scales_host, &mut scales).unwrap();
        s.memcpy_htod(&[0, 1, 2, 2, 1, 0, 2, 0], &mut ids).unwrap();
        let mut reused = HalfMirror::new(&s, 8, cols).unwrap();
        let pointers = (
            reused.half.device_ptr(&s).0,
            reused.scale.device_ptr(&s).0,
            reused.status.device_ptr(&s).0,
        );
        for rows in [8, 1, 5, 8] {
            reused
                .gather(&s, &codes, &scales, Some(&ids), rows)
                .unwrap();
            let mut fresh = HalfMirror::new(&s, rows, cols).unwrap();
            fresh.gather(&s, &codes, &scales, Some(&ids), rows).unwrap();
            let mut a = vec![0u8; rows * cols * 2];
            let mut b = vec![0u8; a.len()];
            s.memcpy_dtoh(&reused.half.slice(..a.len()), &mut a)
                .unwrap();
            s.memcpy_dtoh(&fresh.half, &mut b).unwrap();
            s.synchronize().unwrap();
            assert_eq!(a, b, "reused half values rows={rows}");
            let mut sa = vec![0f32; rows];
            let mut sb = vec![0f32; rows];
            s.memcpy_dtoh(&reused.scale.slice(..rows), &mut sa).unwrap();
            s.memcpy_dtoh(&fresh.scale, &mut sb).unwrap();
            s.synchronize().unwrap();
            assert_eq!(sa, sb, "reused scales rows={rows}");
            assert_eq!(
                pointers,
                (
                    reused.half.device_ptr(&s).0,
                    reused.scale.device_ptr(&s).0,
                    reused.status.device_ptr(&s).0
                )
            );
        }
        codes_host[2 * cols] = 127;
        s.memcpy_htod(&codes_host, &mut codes).unwrap();
        assert!(
            reused
                .gather(&s, &codes, &scales, Some(&ids), 8)
                .unwrap_err()
                .contains("not lossless")
        );
        // Source row 2 is outside this smaller live result; stale bad status must not reject it.
        reused.gather(&s, &codes, &scales, Some(&ids), 1).unwrap();
        codes_host[2 * cols] = 0;
        s.memcpy_htod(&codes_host, &mut codes).unwrap();
        reused.gather(&s, &codes, &scales, Some(&ids), 8).unwrap();
        assert!(reused.gather(&s, &codes, &scales, Some(&ids), 9).is_err());
        let work = GroupedWork::new(&s, 256, 6, 4096, 2048).unwrap();
        let actual = work.routes.bytes
            + (work.input.half.len() + work.intermediate.half.len()) as u64
            + ((work.input.scale.len()
                + work.input.status.len()
                + work.intermediate.scale.len()
                + work.intermediate.status.len()
                + work.contribution.len())
                * 4) as u64;
        assert_eq!(work.bytes, actual);
        assert_eq!(work.bytes, 175352);
        s.synchronize().unwrap();
        // Exercise the mutable pointer API after reuse as well; no new allocation is made.
        assert_eq!(reused.half.device_ptr_mut(&s).0, pointers.0);
        println!(
            "PASS reusable matrix mirrors: shrink/grow, fresh equivalence, stable pointers, invalid/stale status and byte accounting"
        );
    }

    #[test]
    fn matrix_program_and_state_are_explicit() {
        assert_eq!(resolve_program(None), Ok(false));
        assert_eq!(resolve_program(Some("reference")), Ok(false));
        assert_eq!(resolve_program(Some("matrix")), Ok(true));
        for raw in ["", "1", "MATRIX", " matrix"] {
            assert!(resolve_program(Some(raw)).is_err());
        }
        for runtime in [false, true] {
            for state in [false, true] {
                assert_eq!(ensure_program(runtime, state).is_ok(), runtime == state);
            }
        }
    }
    #[test]
    fn route_policy_is_explicit() {
        assert_eq!(resolve(None), Ok(false));
        assert_eq!(resolve(Some("host")), Ok(false));
        assert_eq!(resolve(Some("device")), Ok(true));
        for raw in ["1", "DEVICE", " device", ""] {
            assert!(resolve(Some(raw)).is_err());
        }
    }
}
#[cfg(test)]
mod half2_chain_identity_tests {
    use super::{GroupedWork, modelopt_table};
    use crate::dsv4_ep::{EpCompute, EpScratch};
    use crate::dsv4_ffi as k;
    use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

    struct GateRestore;

    impl Drop for GateRestore {
        fn drop(&mut self) {
            crate::clear_moe_f16g_gu_fuse_for_gate();
            crate::clear_moe_f16g_m1_tc_for_gate();
            crate::clear_moe_f16g_gu_m1_tc_for_gate();
            crate::clear_moe_f16g_gu_half2_for_gate();
            crate::clear_moe_f16g_down_m1_half2_for_gate();
        }
    }

    struct ChainResult {
        h: Vec<f32>,
        pairs: Vec<i32>,
        contribution: Vec<f32>,
    }

    fn view(x: &mut EpScratch) -> EpCompute<'_> {
        EpCompute {
            xq: &x.xq,
            xs: &x.xs,
            ids: &x.ids,
            weights: &x.weights,
            g1: &mut x.g1,
            g3: &mut x.g3,
            h: &mut x.h,
            hq: &mut x.hq,
            hs: &mut x.hs,
            contribution: &mut x.contribution,
        }
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn run_chain(
        gpu: &memra_runtime::Gpu,
        work: &mut GroupedWork,
        scratch: &mut EpScratch,
        table: &CudaSlice<u64>,
        x: &CudaSlice<f32>,
        scale2: &CudaSlice<f32>,
        scale2_host: &[f32],
        selected: &[i32],
        route_weights: &[f32],
        hidden: usize,
        inter: usize,
        topk: usize,
    ) -> ChainResult {
        let s = gpu.stream();
        let slots = selected.len();
        assert_eq!(slots, topk);
        s.memcpy_htod(selected, &mut scratch.ids.slice_mut(..slots))
            .unwrap();
        s.memcpy_htod(route_weights, &mut scratch.weights.slice_mut(..slots))
            .unwrap();
        unsafe {
            k::ck(
                "half2 chain fixture FP8 input",
                k::memra_dsv4_act_quant_fp8(
                    x.device_ptr(&s).0 as *const f32,
                    scratch.xq.device_ptr_mut(&s).0 as *mut std::ffi::c_void,
                    scratch.xs.device_ptr_mut(&s).0 as *mut f32,
                    1,
                    hidden as i32,
                    s.cu_stream().cast(),
                ),
            )
            .unwrap();
        }
        work.prepare(gpu, &view(scratch), scale2, scale2_host, 1, topk, true)
            .unwrap();
        work.set_gu_fuse_for_plain(true);
        work.gate_up(gpu, table, &mut view(scratch), 6.0).unwrap();
        let live = work.routes.live_slots;
        let h = s.clone_dtoh(&scratch.h.slice(..live * inter)).unwrap();
        let pairs = s.clone_dtoh(&work.routes.pairs.slice(..live)).unwrap();
        work.down(gpu, table, &mut view(scratch)).unwrap();
        let contribution = s
            .clone_dtoh(&scratch.contribution.slice(..slots * hidden))
            .unwrap();
        s.synchronize().unwrap();
        ChainResult {
            h,
            pairs,
            contribution,
        }
    }

    fn map_h_bits(h: &[f32], pairs: &[i32], slots: usize, inter: usize) -> Vec<u32> {
        assert_eq!(h.len(), pairs.len() * inter);
        let mut mapped = vec![0u32; slots * inter];
        let mut seen = vec![false; slots];
        for (row, &pair) in pairs.iter().enumerate() {
            let slot = usize::try_from(pair).expect("non-negative grouped pair");
            assert!(slot < slots);
            assert!(!seen[slot], "duplicate original slot");
            seen[slot] = true;
            mapped[slot * inter..(slot + 1) * inter]
                .copy_from_slice(&bits(&h[row * inter..(row + 1) * inter]));
        }
        assert!(
            seen.into_iter().all(|value| value),
            "partition missed a slot"
        );
        mapped
    }

    fn merge_h_bits(
        a: &[f32],
        a_pairs: &[i32],
        b: &[f32],
        b_pairs: &[i32],
        slots: usize,
        inter: usize,
    ) -> Vec<u32> {
        assert_eq!(a.len(), a_pairs.len() * inter);
        assert_eq!(b.len(), b_pairs.len() * inter);
        let mut mapped = vec![0u32; slots * inter];
        let mut seen = vec![false; slots];
        for (values, pairs) in [(a, a_pairs), (b, b_pairs)] {
            for (row, &pair) in pairs.iter().enumerate() {
                let slot = usize::try_from(pair).expect("non-negative grouped pair");
                assert!(slot < slots);
                assert!(!seen[slot], "partition H overlap");
                seen[slot] = true;
                mapped[slot * inter..(slot + 1) * inter]
                    .copy_from_slice(&bits(&values[row * inter..(row + 1) * inter]));
            }
        }
        assert!(
            seen.into_iter().all(|value| value),
            "partition missed a slot"
        );
        mapped
    }

    #[test]
    #[ignore = "requires CUDA; run under the rig lock; full ModelOpt GU/down half2 identity only"]
    fn cuda_half2_chain_identity() {
        let _restore = GateRestore;
        assert!(crate::moe_f16g_mode() >= 2);
        assert!(crate::moe_f16g_direct_on(crate::QT_NVFP4_MODELOPT));
        assert!(crate::moe_f16g_tail_on());

        let gpu = memra_runtime::Gpu::new(0).unwrap();
        let s = gpu.stream();
        let (ne, hidden, inter, topk) = (16, 4096, 2048, 6);
        let slots = topk;
        let wb = hidden * inter / 2;
        let sb = hidden * inter / 16;
        let weight_data: Vec<u8> = (0..ne * 3 * wb)
            .map(|i| {
                let expert = i / (3 * wb);
                let projection = (i / wb) % 3;
                ((i * 37 + 17 + expert * 13 + projection * 59) % 256) as u8
            })
            .collect();
        let scale_data: Vec<u8> = (0..ne * 3 * sb)
            .map(|i| {
                let expert = i / (3 * sb);
                let projection = (i / sb) % 3;
                ((5 + (i + expert * 7 + projection * 3) % 4) << 3) as u8
            })
            .collect();
        let weights = s.clone_htod(&weight_data).unwrap();
        let scales = s.clone_htod(&scale_data).unwrap();
        let full_table = modelopt_table(&s, &weights, &scales, ne, hidden, inter).unwrap();

        let mut shard_w = Vec::new();
        let mut shard_s = Vec::new();
        let mut shard_tables = Vec::new();
        for first in [0, ne / 2] {
            let mut w = s.alloc_zeros::<u8>(ne / 2 * 3 * wb).unwrap();
            let mut sc = s.alloc_zeros::<u8>(ne / 2 * 3 * sb).unwrap();
            s.memcpy_dtod(
                &weights.slice(first * 3 * wb..(first + ne / 2) * 3 * wb),
                &mut w,
            )
            .unwrap();
            s.memcpy_dtod(
                &scales.slice(first * 3 * sb..(first + ne / 2) * 3 * sb),
                &mut sc,
            )
            .unwrap();
            shard_tables.push(modelopt_table(&s, &w, &sc, ne / 2, hidden, inter).unwrap());
            shard_w.push(w);
            shard_s.push(sc);
        }
        let scale2_host: Vec<f32> = (0..ne * 3)
            .map(|i| 2.0_f32.powi((i % 7) as i32 - 4))
            .collect();
        let scale2 = s.clone_htod(&scale2_host).unwrap();
        let input: Vec<f32> = (0..hidden)
            .map(|i| ((i % 31) as f32 - 15.0) / 16.0)
            .collect();
        let x = s.clone_htod(&input).unwrap();
        let selected = [0, 8, 1, 9, 2, 10];
        let route_weights: Vec<f32> = (0..slots).map(|i| (i + 1) as f32 / 32.0).collect();
        let mut full = EpScratch::new(&gpu, &gpu, 1, topk, hidden, inter, None).unwrap();
        let mut full_candidate = EpScratch::new(&gpu, &gpu, 1, topk, hidden, inter, None).unwrap();
        let mut part_a = EpScratch::new(&gpu, &gpu, 1, topk, hidden, inter, None).unwrap();
        let mut part_b = EpScratch::new(&gpu, &gpu, 1, topk, hidden, inter, None).unwrap();
        let mut full_work = GroupedWork::new(&s, ne, slots, hidden, inter).unwrap();
        let mut candidate_work = GroupedWork::new(&s, ne, slots, hidden, inter).unwrap();
        let mut a_work =
            GroupedWork::new_partition(&s, ne, 0, ne / 2, slots, hidden, inter).unwrap();
        let mut b_work =
            GroupedWork::new_partition(&s, ne, ne / 2, ne / 2, slots, hidden, inter).unwrap();

        // Baseline: GU fuse + regular m_e=1 down; both new packed-half2 doors off.
        crate::set_moe_f16g_gu_fuse_for_gate(true);
        crate::set_moe_f16g_m1_tc_for_gate(true);
        crate::set_moe_f16g_gu_m1_tc_for_gate(false);
        crate::set_moe_f16g_gu_half2_for_gate(false);
        crate::set_moe_f16g_down_m1_half2_for_gate(false);
        let baseline = run_chain(
            &gpu,
            &mut full_work,
            &mut full,
            &full_table,
            &x,
            &scale2,
            &scale2_host,
            &selected,
            &route_weights,
            hidden,
            inter,
            topk,
        );

        let gu_before = crate::moe_f16g_gu_half2_dispatches();
        let down_before = crate::moe_f16g_down_m1_half2_dispatches();
        crate::set_moe_f16g_gu_half2_for_gate(true);
        crate::set_moe_f16g_down_m1_half2_for_gate(true);
        let candidate = run_chain(
            &gpu,
            &mut candidate_work,
            &mut full_candidate,
            &full_table,
            &x,
            &scale2,
            &scale2_host,
            &selected,
            &route_weights,
            hidden,
            inter,
            topk,
        );
        let gu_after = crate::moe_f16g_gu_half2_dispatches();
        let down_after = crate::moe_f16g_down_m1_half2_dispatches();
        assert!(gu_after > gu_before, "GU half2 launcher did not engage");
        assert!(
            down_after > down_before,
            "down m1 half2 launcher did not engage"
        );
        assert_eq!(
            bits(&baseline.h),
            bits(&candidate.h),
            "full-bank GU H identity"
        );
        assert_eq!(
            bits(&baseline.contribution),
            bits(&candidate.contribution),
            "full-bank contribution identity"
        );

        let a = run_chain(
            &gpu,
            &mut a_work,
            &mut part_a,
            &shard_tables[0],
            &x,
            &scale2,
            &scale2_host,
            &selected,
            &route_weights,
            hidden,
            inter,
            topk,
        );
        let b = run_chain(
            &gpu,
            &mut b_work,
            &mut part_b,
            &shard_tables[1],
            &x,
            &scale2,
            &scale2_host,
            &selected,
            &route_weights,
            hidden,
            inter,
            topk,
        );
        unsafe {
            k::ck(
                "half2 partition original-slot merge",
                k::memra_dsv4_ep_merge_slots(
                    part_a.contribution.device_ptr_mut(&s).0 as *mut f32,
                    part_b.contribution.device_ptr(&s).0 as *const f32,
                    part_a.ids.device_ptr(&s).0 as *const i32,
                    slots as i32,
                    hidden as i32,
                    (ne / 2) as i32,
                    (ne / 2) as i32,
                    s.cu_stream().cast(),
                ),
            )
            .unwrap();
        }
        let merged = s
            .clone_dtoh(&part_a.contribution.slice(..slots * hidden))
            .unwrap();
        s.synchronize().unwrap();
        assert_eq!(a.pairs.len() + b.pairs.len(), slots);
        assert_eq!(
            map_h_bits(&candidate.h, &candidate.pairs, slots, inter),
            merge_h_bits(&a.h, &a.pairs, &b.h, &b.pairs, slots, inter),
            "partitioned GU H identity"
        );
        assert!(
            bits(&candidate.contribution) == bits(&merged),
            "partitioned contribution identity: first mismatch {:?}",
            candidate
                .contribution
                .iter()
                .zip(&merged)
                .enumerate()
                .find(|(_, (expected, actual))| expected.to_bits() != actual.to_bits())
        );
        println!(
            "PASS GU/down half2 complete-chain identity: h_bits={} contribution_bits={} gu_half2_dispatches={} down_m1_half2_dispatches={}",
            candidate.h.len(),
            candidate.contribution.len(),
            gu_after - gu_before,
            down_after - down_before
        );
        drop((shard_tables, shard_w, shard_s));
    }
}
