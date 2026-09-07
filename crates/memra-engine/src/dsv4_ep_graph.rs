//! Gate-only multi-device graph for the matrix EP plain path.
//!
//! The graph is deliberately a parent graph with per-device child graphs.  The
//! owner and peer expert halves are captured on their own streams; the parent
//! owns the explicit owner->peer and peer->owner copies and the fork/join edges.
//! This is the shape that can include the real matrix EP join without trying to
//! capture `cudaMemcpyPeerAsync` itself.

use crate::dsv4_ep::{EpCompute, EpLayer, EpScratch};
use crate::dsv4_ffi as k;
use crate::dsv4_grouped::GroupedWork;
use cudarc::driver::{CudaGraph, CudaSlice, DevicePtr, DevicePtrMut};
use memra_runtime::Gpu;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

type Res<T> = Result<T, String>;

static MATRIX_EP_GRAPH_FOR_GATE: AtomicBool = AtomicBool::new(false);
static MATRIX_EP_GRAPH_DISPATCHES: AtomicU64 = AtomicU64::new(0);

/// Gate-only process switch.  It has no environment reader or serving default.
pub(crate) fn set_for_gate(enabled: bool) -> bool {
    MATRIX_EP_GRAPH_FOR_GATE.swap(enabled, Ordering::AcqRel)
}

pub(crate) fn enabled() -> bool {
    MATRIX_EP_GRAPH_FOR_GATE.load(Ordering::Acquire)
}

pub(crate) fn dispatches() -> u64 {
    MATRIX_EP_GRAPH_DISPATCHES.load(Ordering::Acquire)
}

#[derive(Debug, Clone, Copy)]
struct CopySpec {
    src: cudarc::driver::sys::CUdeviceptr,
    dst: cudarc::driver::sys::CUdeviceptr,
    graph_ctx: cudarc::driver::sys::CUcontext,
    bytes: usize,
}

/// Parent graph plus the child graphs it references.  Child graphs must stay
/// alive until the parent exec is destroyed: CUDA clones their topology into
/// the parent at AddChildGraphNode time, but the retained handles are kept here
/// to preserve the allocation and module lifetime contract explicitly.
pub(crate) struct MatrixEpGraph {
    exec: cudarc::driver::sys::CUgraphExec,
    parent: cudarc::driver::sys::CUgraph,
    _children: Vec<CudaGraph>,
}

// The graph is owned by one decode state and launched by the decode thread.
// CUDA graph handles are process handles, not Rust-thread-bound references.
unsafe impl Send for MatrixEpGraph {}

impl Drop for MatrixEpGraph {
    fn drop(&mut self) {
        unsafe {
            let _ = cudarc::driver::sys::cuGraphExecDestroy(self.exec);
            let _ = cudarc::driver::sys::cuGraphDestroy(self.parent);
        }
    }
}

impl MatrixEpGraph {
    pub(crate) fn launch(&self, owner: &Gpu) -> Res<()> {
        owner.ctx.bind_to_thread().map_err(|e| e.to_string())?;
        let rc = unsafe {
            cudarc::driver::sys::cuGraphLaunch(
                self.exec,
                owner.stream().cu_stream() as cudarc::driver::sys::CUstream,
            )
        };
        if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(format!("matrix EP graph launch: {rc:?}"));
        }
        MATRIX_EP_GRAPH_DISPATCHES.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }
}

/// Empty/ready/fail-latched state for one model layer.  A refusal must not
/// recapture every token while the request continues on the eager twin.
pub(crate) enum MatrixEpGraphSlot {
    Empty,
    Ready(MatrixEpGraph),
    Refused,
}

impl MatrixEpGraphSlot {
    pub(crate) fn empty() -> Self {
        Self::Empty
    }
}

fn capture_child(gpu: &Gpu, mut body: impl FnMut() -> Res<()>) -> Res<CudaGraph> {
    use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};

    if gpu.ctx.is_event_tracking() {
        return Err("matrix EP graph requires event tracking disabled before allocation".into());
    }
    gpu.ctx.bind_to_thread().map_err(|e| e.to_string())?;
    let stream = gpu.stream();
    stream.synchronize().map_err(|e| e.to_string())?;
    stream
        .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        .map_err(|e| format!("matrix EP graph begin capture: {e}"))?;
    let result = body();
    let ended = stream
        .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
    result?;
    let graph = ended
        .map_err(|e| format!("matrix EP graph end capture: {e}"))?
        .ok_or("matrix EP graph produced no child graph")?;
    graph
        .upload()
        .map_err(|e| format!("matrix EP graph child upload: {e}"))?;
    Ok(graph)
}

fn add_child(
    parent: cudarc::driver::sys::CUgraph,
    deps: &[cudarc::driver::sys::CUgraphNode],
    child: &CudaGraph,
) -> Res<cudarc::driver::sys::CUgraphNode> {
    let mut node = std::ptr::null_mut();
    let rc = unsafe {
        cudarc::driver::sys::cuGraphAddChildGraphNode(
            &mut node,
            parent,
            if deps.is_empty() {
                std::ptr::null()
            } else {
                deps.as_ptr()
            },
            deps.len(),
            child.cu_graph(),
        )
    };
    if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        return Err(format!("matrix EP graph add child: {rc:?}"));
    }
    Ok(node)
}

fn add_copy(
    parent: cudarc::driver::sys::CUgraph,
    deps: &[cudarc::driver::sys::CUgraphNode],
    spec: CopySpec,
) -> Res<cudarc::driver::sys::CUgraphNode> {
    let copy = cudarc::driver::sys::CUDA_MEMCPY3D_st {
        srcXInBytes: 0,
        srcY: 0,
        srcZ: 0,
        srcLOD: 0,
        srcMemoryType: cudarc::driver::sys::CUmemorytype_enum::CU_MEMORYTYPE_DEVICE,
        srcHost: std::ptr::null(),
        srcDevice: spec.src,
        srcArray: std::ptr::null_mut(),
        reserved0: std::ptr::null_mut(),
        srcPitch: spec.bytes,
        srcHeight: 1,
        dstXInBytes: 0,
        dstY: 0,
        dstZ: 0,
        dstLOD: 0,
        dstMemoryType: cudarc::driver::sys::CUmemorytype_enum::CU_MEMORYTYPE_DEVICE,
        dstHost: std::ptr::null_mut(),
        dstDevice: spec.dst,
        dstArray: std::ptr::null_mut(),
        reserved1: std::ptr::null_mut(),
        dstPitch: spec.bytes,
        dstHeight: 1,
        WidthInBytes: spec.bytes,
        Height: 1,
        Depth: 1,
    };
    let mut node = std::ptr::null_mut();
    let rc = unsafe {
        cudarc::driver::sys::cuGraphAddMemcpyNode(
            &mut node,
            parent,
            if deps.is_empty() {
                std::ptr::null()
            } else {
                deps.as_ptr()
            },
            deps.len(),
            &copy,
            spec.graph_ctx,
        )
    };
    if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        return Err(format!("matrix EP graph add P2P copy: {rc:?}"));
    }
    Ok(node)
}

fn add_parent(
    owner_child: CudaGraph,
    peer_child: CudaGraph,
    post_child: CudaGraph,
    copies: [CopySpec; 5],
) -> Res<MatrixEpGraph> {
    use cudarc::driver::sys;
    let mut parent = std::ptr::null_mut();
    let rc = unsafe { sys::cuGraphCreate(&mut parent, 0) };
    if rc != sys::CUresult::CUDA_SUCCESS {
        return Err(format!("matrix EP graph create parent: {rc:?}"));
    }
    let result = (|| -> Res<MatrixEpGraph> {
        let owner_node = add_child(parent, &[], &owner_child)?;
        let tx0 = add_copy(parent, &[], copies[0])?;
        let tx1 = add_copy(parent, &[], copies[1])?;
        let tx2 = add_copy(parent, &[], copies[2])?;
        let tx3 = add_copy(parent, &[], copies[3])?;
        let tx = [tx0, tx1, tx2, tx3];
        let peer_node = add_child(parent, &tx, &peer_child)?;
        let rx_node = add_copy(parent, &[peer_node], copies[4])?;
        let post_deps = [owner_node, rx_node];
        let _post_node = add_child(parent, &post_deps, &post_child)?;
        let mut exec = std::ptr::null_mut();
        let rc = unsafe { sys::cuGraphInstantiateWithFlags(&mut exec, parent, 0) };
        if rc != sys::CUresult::CUDA_SUCCESS {
            return Err(format!("matrix EP graph instantiate: {rc:?}"));
        }
        Ok(MatrixEpGraph {
            exec,
            parent,
            _children: vec![owner_child, peer_child, post_child],
        })
    })();
    if result.is_err() {
        unsafe {
            let _ = sys::cuGraphDestroy(parent);
        }
    }
    result
}

struct MergeSpec {
    local_contribution: cudarc::driver::sys::CUdeviceptr,
    returned: cudarc::driver::sys::CUdeviceptr,
    ids: cudarc::driver::sys::CUdeviceptr,
    slots: i32,
    hidden: i32,
    peer_first: i32,
    count: i32,
    stream: *mut c_void,
}

/// Build phase for the matrix EP graph.  The owner/peer `EpCompute` borrows are
/// confined to this method; that lets the caller capture the owner FFN tail
/// after the expert workspaces have been released.
pub(crate) struct MatrixEpGraphBuilder {
    owner: *const Gpu,
    owner_child: CudaGraph,
    peer_child: CudaGraph,
    copies: [CopySpec; 5],
    merge: MergeSpec,
}

impl MatrixEpGraphBuilder {
    pub(crate) fn capture_expert(
        owner: &Gpu,
        peer: &Gpu,
        bank: &EpLayer,
        table: &CudaSlice<u64>,
        scale2: &CudaSlice<f32>,
        scale2_host: &[f32],
        local: &mut EpCompute<'_>,
        local_work: &mut GroupedWork,
        remote: &mut EpScratch,
        rows: usize,
        topk: usize,
        hidden: usize,
        limit: f32,
        allow_gu_fuse: bool,
    ) -> Res<Self> {
        if rows != 1 || topk == 0 || hidden == 0 {
            return Err("matrix EP graph currently supports only non-empty t=1 decode".into());
        }
        if crate::dsv4_grouped::route_validation_enabled()
            || crate::dsv4_grouped::mirror_validation_enabled()
        {
            return Err("matrix EP graph requires route and mirror validation disabled".into());
        }
        let peer_table = bank
            .peer_table
            .as_ref()
            .ok_or("matrix EP peer table missing")?;
        let global = bank
            .count
            .checked_mul(2)
            .ok_or("matrix EP graph expert count overflow")?;
        let slots = rows * topk;
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
            return Err("matrix EP graph ownership mismatch".into());
        }
        local_work.set_gu_fuse_for_plain(allow_gu_fuse);
        if let Some(work) = remote.grouped.as_mut() {
            work.set_gu_fuse_for_plain(allow_gu_fuse);
        } else {
            return Err("matrix EP graph peer grouped workspace missing".into());
        }

        let owner_child = capture_child(owner, || {
            local_work.prepare(owner, local, scale2, scale2_host, rows, topk, true)?;
            local_work.gate_up(owner, table, local, limit)?;
            local_work.down(owner, table, local)?;
            Ok(())
        })?;

        let peer_child = {
            let peer_work = remote
                .grouped
                .as_mut()
                .ok_or("matrix EP graph peer grouped workspace missing")?;
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
            capture_child(peer, || {
                peer_work.prepare(
                    peer,
                    &peer_compute,
                    &bank.peer_s2,
                    scale2_host,
                    rows,
                    topk,
                    true,
                )?;
                peer_work.gate_up(peer, peer_table, &mut peer_compute, limit)?;
                peer_work.down(peer, peer_table, &mut peer_compute)?;
                Ok(())
            })?
        };

        let os = owner.stream();
        let ps = peer.stream();
        let owner_ctx = owner.ctx.cu_ctx();
        let peer_ctx = peer.ctx.cu_ctx();
        let ptr = |s: &cudarc::driver::CudaStream, x: &CudaSlice<u8>| x.device_ptr(s).0;
        let (src_xq, src_xs, src_ids, src_weights) = (
            ptr(&os, local.xq),
            local.xs.device_ptr(&os).0,
            local.ids.device_ptr(&os).0,
            local.weights.device_ptr(&os).0,
        );
        let (dst_xq, dst_xs, dst_ids, dst_weights) = (
            remote.xq.device_ptr_mut(&ps).0,
            remote.xs.device_ptr_mut(&ps).0,
            remote.ids.device_ptr_mut(&ps).0,
            remote.weights.device_ptr_mut(&ps).0,
        );
        let bytes_xq = rows * hidden;
        let bytes_xs = rows * hidden / 128 * std::mem::size_of::<f32>();
        let bytes_slots = slots * std::mem::size_of::<i32>();
        let bytes_weights = slots * std::mem::size_of::<f32>();
        let bytes_contribution = slots * hidden * std::mem::size_of::<f32>();
        let (rx_src, rx_dst) = (
            remote.contribution.device_ptr(&ps).0,
            remote.returned.device_ptr_mut(&os).0,
        );
        let copies = [
            CopySpec {
                src: src_xq,
                dst: dst_xq,
                graph_ctx: owner_ctx,
                bytes: bytes_xq,
            },
            CopySpec {
                src: src_xs,
                dst: dst_xs,
                graph_ctx: owner_ctx,
                bytes: bytes_xs,
            },
            CopySpec {
                src: src_ids,
                dst: dst_ids,
                graph_ctx: owner_ctx,
                bytes: bytes_slots,
            },
            CopySpec {
                src: src_weights,
                dst: dst_weights,
                graph_ctx: owner_ctx,
                bytes: bytes_weights,
            },
            CopySpec {
                src: rx_src,
                dst: rx_dst,
                graph_ctx: peer_ctx,
                bytes: bytes_contribution,
            },
        ];

        Ok(Self {
            owner,
            owner_child,
            peer_child,
            copies,
            merge: MergeSpec {
                local_contribution: local.contribution.device_ptr_mut(&os).0,
                returned: remote.returned.device_ptr(&os).0,
                ids: local.ids.device_ptr(&os).0,
                slots: slots as i32,
                hidden: hidden as i32,
                peer_first: bank.peer_first as i32,
                count: bank.count as i32,
                stream: os.cu_stream().cast(),
            },
        })
    }

    /// Finish the parent graph after the owner-side FFN tail closure is ready.
    pub(crate) fn finish<F>(self, mut post: F) -> Res<MatrixEpGraph>
    where
        F: FnMut() -> Res<()>,
    {
        let owner = unsafe { &*self.owner };
        let owner_post = capture_child(owner, || {
            unsafe {
                k::ck(
                    "matrix EP graph merge",
                    k::memra_dsv4_ep_merge_slots(
                        self.merge.local_contribution as *mut f32,
                        self.merge.returned as *const f32,
                        self.merge.ids as *const i32,
                        self.merge.slots,
                        self.merge.hidden,
                        self.merge.peer_first,
                        self.merge.count,
                        self.merge.stream,
                    ),
                )?;
            }
            post()
        })?;
        add_parent(self.owner_child, self.peer_child, owner_post, self.copies)
    }
}
