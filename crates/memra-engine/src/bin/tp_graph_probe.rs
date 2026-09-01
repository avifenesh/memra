//! Multi-context CUDA-graph capture probe for the step TP decode chain.
//!
//! The v2 decode drivers order every cross-stream seam with events and every fork rejoins the
//! model engine's stream (e -> ranks -> root -> e). Stream capture pulls event-dependent
//! streams into the capture, so the WHOLE chain should be capturable as one graph — this probe
//! retires exactly that mechanism risk before any driver capture work: build a miniature
//! e/rank0/rank1 evented chain (peer copies + kernels + reduce), capture it on e's stream,
//! then relaunch with fresh inputs and verify the outputs track.
//!
//! PASS = capture succeeds AND three replays with distinct inputs produce the expected bytes.

use cudarc::driver::{CudaEvent, CudaSlice};
use memra_engine::Engine;

const N: usize = 4096;

struct ProbeBufs {
    input_e: CudaSlice<f32>,
    out_e: CudaSlice<f32>,
    in0: CudaSlice<f32>,
    part0: CudaSlice<f32>,
    in1: CudaSlice<f32>,
    part1: CudaSlice<f32>,
    peer1: CudaSlice<f32>,
    reduced: CudaSlice<f32>,
}

struct ProbeEvents {
    ev_e: CudaEvent,
    ev0: CudaEvent,
    ev1: CudaEvent,
    ev_root: CudaEvent,
    ev_done: CudaEvent,
}

/// The miniature v2 chain: e -> (rank0, rank1) -> root -> e, all evented.
/// rank r computes part_r = in_r * (2 + r); root reduces part0 + part1; e copies out.
/// Expected: out = input * 5.
fn chain(
    e: &Engine,
    rank0: &Engine,
    rank1: &Engine,
    bufs: &mut ProbeBufs,
    events: &ProbeEvents,
) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("[tp-graph-probe] chain: entry record");
    {
        let _main = e.gpu.enter_main()?;
        events.ev_e.record(&e.stream())?;
    }
    for rank in 0..2usize {
        eprintln!("[tp-graph-probe] chain: rank {rank} section");
        let (engine, scale, ev) = if rank == 0 {
            (rank0, 2.0f32, &events.ev0)
        } else {
            (rank1, 3.0f32, &events.ev1)
        };
        let (input, part) = if rank == 0 {
            (&mut bufs.in0, &mut bufs.part0)
        } else {
            (&mut bufs.in1, &mut bufs.part1)
        };
        let _main = engine.gpu.enter_main()?;
        engine.stream().wait(&events.ev_e)?;
        {
            let mut dst = input.slice_mut(0..N);
            engine
                .stream()
                .memcpy_dtod(&bufs.input_e.slice(0..N), &mut dst)?;
        }
        engine.scale_inplace(input, scale, N)?;
        {
            let mut dst = part.slice_mut(0..N);
            engine.stream().memcpy_dtod(&input.slice(0..N), &mut dst)?;
        }
        ev.record(&engine.stream())?;
    }
    eprintln!("[tp-graph-probe] chain: root section");
    {
        let _main = rank0.gpu.enter_main()?;
        rank0.stream().wait(&events.ev0)?;
        rank0.stream().wait(&events.ev1)?;
        {
            let mut dst = bufs.peer1.slice_mut(0..N);
            rank0
                .stream()
                .memcpy_dtod(&bufs.part1.slice(0..N), &mut dst)?;
        }
        let ProbeBufs {
            part0,
            peer1,
            reduced,
            ..
        } = bufs;
        rank0.add(part0, peer1, reduced, N)?;
        events.ev_root.record(&rank0.stream())?;
    }
    eprintln!("[tp-graph-probe] chain: e-out section");
    {
        let _main = e.gpu.enter_main()?;
        e.stream().wait(&events.ev_root)?;
        {
            let mut dst = bufs.out_e.slice_mut(0..N);
            e.stream()
                .memcpy_dtod(&bufs.reduced.slice(0..N), &mut dst)?;
        }
        events.ev_done.record(&e.stream())?;
        e.stream().wait(&events.ev_done)?;
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("[tp-graph-probe] FAIL: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = memra_engine::tp::TpE4m3HostBounce::new_native_p2p(&[0, 1])?;
    let e = Engine::new(0)?;
    let rank0 = runtime.rank_engine(0).ok_or("no rank 0")?;
    let rank1 = runtime.rank_engine(1).ok_or("no rank 1")?;

    let mut bufs = {
        let (input_e, out_e) = {
            let _main = e.gpu.enter_main()?;
            (e.htod(&vec![0.0f32; N])?, e.htod(&vec![0.0f32; N])?)
        };
        let (in0, part0, peer1, reduced) = {
            let _main = rank0.gpu.enter_main()?;
            (
                rank0.htod(&vec![0.0f32; N])?,
                rank0.htod(&vec![0.0f32; N])?,
                rank0.htod(&vec![0.0f32; N])?,
                rank0.htod(&vec![0.0f32; N])?,
            )
        };
        let (in1, part1) = {
            let _main = rank1.gpu.enter_main()?;
            (rank1.htod(&vec![0.0f32; N])?, rank1.htod(&vec![0.0f32; N])?)
        };
        ProbeBufs {
            input_e,
            out_e,
            in0,
            part0,
            in1,
            part1,
            peer1,
            reduced,
        }
    };
    let events = {
        let (ev_e, ev_done) = {
            let _main = e.gpu.enter_main()?;
            (e.ctx().new_event(None)?, e.ctx().new_event(None)?)
        };
        let (ev0, ev_root) = {
            let _main = rank0.gpu.enter_main()?;
            (rank0.ctx().new_event(None)?, rank0.ctx().new_event(None)?)
        };
        let ev1 = {
            let _main = rank1.gpu.enter_main()?;
            rank1.ctx().new_event(None)?
        };
        ProbeEvents {
            ev_e,
            ev0,
            ev1,
            ev_root,
            ev_done,
        }
    };

    // Eager warmup + reference check.
    let seed: Vec<f32> = (0..N).map(|i| (i % 97) as f32 * 0.25 + 1.0).collect();
    {
        let _main = e.gpu.enter_main()?;
        let mut dst = bufs.input_e.slice_mut(0..N);
        e.stream().memcpy_htod(&seed, &mut dst)?;
    }
    chain(&e, rank0, rank1, &mut bufs, &events)?;
    {
        let _main = e.gpu.enter_main()?;
        e.stream().synchronize()?;
        let out = e.dtoh(&bufs.out_e)?;
        let bad = out
            .iter()
            .zip(&seed)
            .filter(|(o, s)| (**o - **s * 5.0).abs() > 1e-3)
            .count();
        if bad != 0 {
            return Err(format!("eager chain wrong: {bad} mismatches").into());
        }
    }
    eprintln!("[tp-graph-probe] eager chain PASS");

    // ATTEMPT 1 (negative result, recorded in the lane notes): whole-chain stream capture
    // fails with CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED at the first CROSS-DEVICE event join —
    // stream capture cannot span devices, and the failed capture leaves the origin stream in a
    // broken capture state (subsequent captures see STREAM_CAPTURE_ISOLATION), so the attempt
    // is not repeated here.

    // ATTEMPT 2: stitched multi-device parent. Children carry NO cross-device event ops —
    // ordering comes from parent dependency edges.
    use cudarc::driver::sys;
    fn cu_try(r: sys::CUresult, what: &str) -> Result<(), Box<dyn std::error::Error>> {
        if r == sys::CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!("{what}: {r:?}").into())
        }
    }

    // Cross-context copies inside captured segments go through RAW cuMemcpyAsync: cudarc's
    // slice-use tracking would insert a safety dependency on the foreign owner stream, which
    // capture rejects (STREAM_CAPTURE_ISOLATION). Pointers cached before capture.
    let raw_ptr = |buf: &CudaSlice<f32>, engine: &Engine| -> u64 {
        use cudarc::driver::DevicePtr;
        let stream = engine.stream();
        let (ptr, _guard) = buf.device_ptr(&stream);
        ptr
    };
    let p_input_e = {
        let _main = e.gpu.enter_main()?;
        raw_ptr(&bufs.input_e, &e)
    };
    let (p_in0, p_part0, p_peer1, p_reduced) = {
        let _main = rank0.gpu.enter_main()?;
        (
            raw_ptr(&bufs.in0, rank0),
            raw_ptr(&bufs.part0, rank0),
            raw_ptr(&bufs.peer1, rank0),
            raw_ptr(&bufs.reduced, rank0),
        )
    };
    let (p_in1, p_part1) = {
        let _main = rank1.gpu.enter_main()?;
        (raw_ptr(&bufs.in1, rank1), raw_ptr(&bufs.part1, rank1))
    };
    let p_out_e = {
        let _main = e.gpu.enter_main()?;
        raw_ptr(&bufs.out_e, &e)
    };
    let _ = (p_part0,);
    let bytes = (N * std::mem::size_of::<f32>()) as usize;
    let raw_copy =
        |dst: u64, src: u64, engine: &Engine| -> Result<(), Box<dyn std::error::Error>> {
            unsafe {
                cu_try(
                    sys::cuMemcpyAsync(
                        dst as sys::CUdeviceptr,
                        src as sys::CUdeviceptr,
                        bytes,
                        engine.stream().cu_stream() as sys::CUstream,
                    ),
                    "cuMemcpyAsync",
                )
            }
        };

    let rank_seg = |engine: &Engine,
                    input: &mut CudaSlice<f32>,
                    p_input: u64,
                    p_src: u64,
                    p_part: u64,
                    scale: f32|
     -> Result<(), Box<dyn std::error::Error>> {
        let _main = engine.gpu.enter_main()?;
        raw_copy(p_input, p_src, engine)?;
        engine.scale_inplace(input, scale, N)?;
        raw_copy(p_part, p_input, engine)?;
        Ok(())
    };

    // Children captured on their own device streams.
    let (c0, _r0) = {
        let ProbeBufs { in0, .. } = &mut bufs;
        let _main = rank0.gpu.enter_main()?;
        rank0.capture_graph_retained(|_| rank_seg(rank0, in0, p_in0, p_input_e, p_part0, 2.0))?
    };
    let (c1, _r1) = {
        let ProbeBufs { in1, .. } = &mut bufs;
        let _main = rank1.gpu.enter_main()?;
        rank1.capture_graph_retained(|_| rank_seg(rank1, in1, p_in1, p_input_e, p_part1, 3.0))?
    };
    let (c2, _r2) = {
        // combine on root (device 0) + handoff toward e's buffer: cross-context transfers via
        // raw copies; the add kernel's operands are all rank0-owned so cudarc launches it.
        let ProbeBufs {
            part0,
            peer1,
            reduced,
            ..
        } = &mut bufs;
        let _main = rank0.gpu.enter_main()?;
        rank0.capture_graph_retained(|_| {
            let _main = rank0.gpu.enter_main()?;
            raw_copy(p_peer1, p_part1, rank0)?;
            rank0.add(part0, peer1, reduced, N)?;
            raw_copy(p_out_e, p_reduced, rank0)?;
            Ok(())
        })?
    };
    eprintln!("[tp-graph-probe] per-device children captured");

    // Parent: c0 and c1 in parallel, c2 depends on both.
    let mut parent: sys::CUgraph = std::ptr::null_mut();
    unsafe {
        cu_try(sys::cuGraphCreate(&mut parent, 0), "cuGraphCreate")?;
    }
    let mut n0: sys::CUgraphNode = std::ptr::null_mut();
    let mut n1: sys::CUgraphNode = std::ptr::null_mut();
    let mut n2: sys::CUgraphNode = std::ptr::null_mut();
    unsafe {
        cu_try(
            sys::cuGraphAddChildGraphNode(&mut n0, parent, std::ptr::null(), 0, c0.cu_graph()),
            "child c0",
        )?;
        cu_try(
            sys::cuGraphAddChildGraphNode(&mut n1, parent, std::ptr::null(), 0, c1.cu_graph()),
            "child c1",
        )?;
        let deps = [n0, n1];
        cu_try(
            sys::cuGraphAddChildGraphNode(&mut n2, parent, deps.as_ptr(), 2, c2.cu_graph()),
            "child c2",
        )?;
    }
    let mut exec: sys::CUgraphExec = std::ptr::null_mut();
    unsafe {
        let _main = e.gpu.enter_main()?;
        cu_try(
            sys::cuGraphInstantiateWithFlags(&mut exec, parent, 0),
            "instantiate parent",
        )?;
    }
    eprintln!("[tp-graph-probe] multi-device parent instantiated");

    for round in 0..3u32 {
        let fresh: Vec<f32> = (0..N)
            .map(|i| ((i + round as usize * 13) % 89) as f32 * 0.5 + 2.0)
            .collect();
        {
            let _main = e.gpu.enter_main()?;
            let mut dst = bufs.input_e.slice_mut(0..N);
            e.stream().memcpy_htod(&fresh, &mut dst)?;
            e.stream().synchronize()?;
        }
        unsafe {
            let _main = e.gpu.enter_main()?;

            let _ = &e; // stream handle below
            cu_try(
                sys::cuGraphLaunch(exec, e.stream().cu_stream() as sys::CUstream),
                "launch parent",
            )?;
        }
        {
            let _main = e.gpu.enter_main()?;
            e.stream().synchronize()?;
        }
        let out = {
            let _main = e.gpu.enter_main()?;
            e.dtoh(&bufs.out_e)?
        };
        let bad = out
            .iter()
            .zip(&fresh)
            .filter(|(o, s)| (**o - **s * 5.0).abs() > 1e-3)
            .count();
        if bad != 0 {
            return Err(format!("stitched replay round {round} wrong: {bad} mismatches").into());
        }
        eprintln!("[tp-graph-probe] stitched replay round {round} PASS");
    }
    eprintln!("[tp-graph-probe] PASS: per-device children + multi-device parent stitching works");

    // M1: per-token param updates inside a CHILD of the instantiated parent — the fa/t_kv
    // requirement. Get c0's cloned child graph off the parent, find its kernel node
    // (scale_f32), rewrite the scale scalar, push with cuGraphExecKernelNodeSetParams_v2 on
    // the PARENT exec, replay, and verify the new scale took.
    {
        let mut child_graph: sys::CUgraph = std::ptr::null_mut();
        unsafe {
            cu_try(
                sys::cuGraphChildGraphNodeGetGraph(n0, &mut child_graph),
                "child GetGraph",
            )?;
        }
        let mut count: usize = 0;
        unsafe {
            cu_try(
                sys::cuGraphGetNodes(child_graph, std::ptr::null_mut(), &mut count),
                "child GetNodes(count)",
            )?;
        }
        let mut nodes: Vec<sys::CUgraphNode> = vec![std::ptr::null_mut(); count];
        unsafe {
            cu_try(
                sys::cuGraphGetNodes(child_graph, nodes.as_mut_ptr(), &mut count),
                "child GetNodes",
            )?;
        }
        nodes.truncate(count);
        let mut kernel_node: Option<sys::CUgraphNode> = None;
        for node in nodes {
            let mut ty = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
            unsafe {
                cu_try(sys::cuGraphNodeGetType(node, &mut ty), "child NodeGetType")?;
            }
            if ty == sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL {
                kernel_node = Some(node);
            }
        }
        let kernel_node = kernel_node.ok_or("no kernel node in child c0")?;
        let mut params: sys::CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
        unsafe {
            cu_try(
                sys::cuGraphKernelNodeGetParams_v2(kernel_node, &mut params),
                "child KernelNodeGetParams",
            )?;
        }
        // scale_f32(y, s, n): rewrite s (param slot 1, f32) 2.0 -> 9.0; expected out = x*12.
        unsafe {
            let slot = *params.kernelParams.add(1) as *mut f32;
            *slot = 9.0;
            cu_try(
                sys::cuGraphExecKernelNodeSetParams_v2(exec, kernel_node, &params),
                "exec child KernelNodeSetParams",
            )?;
        }
        let fresh: Vec<f32> = (0..N).map(|i| (i % 61) as f32 * 0.75 + 1.5).collect();
        {
            let _main = e.gpu.enter_main()?;
            let mut dst = bufs.input_e.slice_mut(0..N);
            e.stream().memcpy_htod(&fresh, &mut dst)?;
            e.stream().synchronize()?;
        }
        unsafe {
            let _main = e.gpu.enter_main()?;
            cu_try(
                sys::cuGraphLaunch(exec, e.stream().cu_stream() as sys::CUstream),
                "launch parent (updated)",
            )?;
        }
        {
            let _main = e.gpu.enter_main()?;
            e.stream().synchronize()?;
        }
        let out = {
            let _main = e.gpu.enter_main()?;
            e.dtoh(&bufs.out_e)?
        };
        let bad = out
            .iter()
            .zip(&fresh)
            .filter(|(o, s)| (**o - **s * 12.0).abs() > 1e-2)
            .count();
        if bad != 0 {
            return Err(format!(
                "child param update did not take: {bad} mismatches (expected x*12)"
            )
            .into());
        }
        eprintln!("[tp-graph-probe] M1 PASS: child kernel param update via parent exec works");
    }

    // M2: capture with cudarc ALLOCS inside (mem-node capture) — the e-glue segments allocate
    // per layer (residuals, matmul outputs); the single-GPU graph lanes prove alloc capture
    // in-tree, this pins it for OUR capture helper + replay-with-frees.
    {
        let _main = e.gpu.enter_main()?;
        let (g_alloc, _r) = e.capture_graph_retained(|e| {
            let mut tmp = e.uninit(N)?;
            {
                let mut dst = tmp.slice_mut(0..N);
                e.stream()
                    .memcpy_dtod(&bufs.input_e.slice(0..N), &mut dst)?;
            }
            e.scale_inplace(&mut tmp, 4.0, N)?;
            {
                let mut dst = bufs.out_e.slice_mut(0..N);
                e.stream().memcpy_dtod(&tmp.slice(0..N), &mut dst)?;
            }
            Ok(())
        })?;
        for round in 0..3u32 {
            let fresh: Vec<f32> = (0..N)
                .map(|i| ((i + round as usize * 7) % 53) as f32 + 1.0)
                .collect();
            {
                let mut dst = bufs.input_e.slice_mut(0..N);
                e.stream().memcpy_htod(&fresh, &mut dst)?;
            }
            g_alloc.launch()?;
            e.stream().synchronize()?;
            let out = e.dtoh(&bufs.out_e)?;
            let bad = out
                .iter()
                .zip(&fresh)
                .filter(|(o, s)| (**o - **s * 4.0).abs() > 1e-3)
                .count();
            if bad != 0 {
                return Err(format!("M2 alloc-capture replay {round}: {bad} mismatches").into());
            }
        }
        eprintln!("[tp-graph-probe] M2 PASS: alloc-inside capture replays correctly");
    }

    // M3: cross-device dep-edge LATENCY. A parent of 2*K alternating-device children (each one
    // tiny kernel) replays as a serial d0->d1->d0->... chain; wall/edge isolates what one
    // cross-device join costs in this driver/topology (the token graph pays ~4/layer).
    {
        use cudarc::driver::sys;
        let k_pairs = 50usize;
        let mut parent: sys::CUgraph = std::ptr::null_mut();
        unsafe {
            cu_try(sys::cuGraphCreate(&mut parent, 0), "M3 cuGraphCreate")?;
        }
        let mut prev: Option<sys::CUgraphNode> = None;
        let mut children = Vec::new();
        for i in 0..(2 * k_pairs) {
            let engine = if i % 2 == 0 { rank0 } else { rank1 };
            let buf = if i % 2 == 0 {
                &mut bufs.in0
            } else {
                &mut bufs.in1
            };
            let (child, _r) = {
                let _main = engine.gpu.enter_main()?;
                engine.capture_graph_retained_nowarm(|_| {
                    engine.scale_inplace(buf, 1.0000001, 64)?;
                    Ok(())
                })?
            };
            let mut node: sys::CUgraphNode = std::ptr::null_mut();
            let deps = prev.map(|p| vec![p]).unwrap_or_default();
            unsafe {
                cu_try(
                    sys::cuGraphAddChildGraphNode(
                        &mut node,
                        parent,
                        if deps.is_empty() {
                            std::ptr::null()
                        } else {
                            deps.as_ptr()
                        },
                        deps.len(),
                        child.cu_graph(),
                    ),
                    "M3 child add",
                )?;
            }
            prev = Some(node);
            children.push(child);
        }
        let mut exec: sys::CUgraphExec = std::ptr::null_mut();
        unsafe {
            let _main = e.gpu.enter_main()?;
            cu_try(
                sys::cuGraphInstantiateWithFlags(&mut exec, parent, 0),
                "M3 instantiate",
            )?;
        }
        // warm + timed replays
        for _ in 0..3 {
            unsafe {
                let _main = e.gpu.enter_main()?;
                cu_try(
                    sys::cuGraphLaunch(exec, e.stream().cu_stream() as sys::CUstream),
                    "M3 launch",
                )?;
            }
        }
        {
            let _main = e.gpu.enter_main()?;
            e.stream().synchronize()?;
        }
        let reps = 20usize;
        let started = std::time::Instant::now();
        for _ in 0..reps {
            unsafe {
                let _main = e.gpu.enter_main()?;
                cu_try(
                    sys::cuGraphLaunch(exec, e.stream().cu_stream() as sys::CUstream),
                    "M3 launch timed",
                )?;
            }
        }
        {
            let _main = e.gpu.enter_main()?;
            e.stream().synchronize()?;
        }
        let wall = started.elapsed().as_secs_f64();
        let per_edge_us = wall / reps as f64 / (2.0 * k_pairs as f64) * 1e6;
        eprintln!(
            "[tp-graph-probe] M3: {} alternating-device children x {reps} replays, \
             wall={:.3}s -> per-edge {:.1}us (kernel ~1us each; the remainder is the \
             cross-device join)",
            2 * k_pairs,
            wall,
            per_edge_us
        );
        // Control: same chain, SAME device (rank0 only) — isolates child-node scheduling.
        let mut parent2: sys::CUgraph = std::ptr::null_mut();
        unsafe {
            cu_try(sys::cuGraphCreate(&mut parent2, 0), "M3 cuGraphCreate2")?;
        }
        let mut prev2: Option<sys::CUgraphNode> = None;
        let mut children2 = Vec::new();
        for _ in 0..(2 * k_pairs) {
            let (child, _r) = {
                let _main = rank0.gpu.enter_main()?;
                rank0.capture_graph_retained_nowarm(|_| {
                    rank0.scale_inplace(&mut bufs.in0, 1.0000001, 64)?;
                    Ok(())
                })?
            };
            let mut node: sys::CUgraphNode = std::ptr::null_mut();
            let deps = prev2.map(|p| vec![p]).unwrap_or_default();
            unsafe {
                cu_try(
                    sys::cuGraphAddChildGraphNode(
                        &mut node,
                        parent2,
                        if deps.is_empty() {
                            std::ptr::null()
                        } else {
                            deps.as_ptr()
                        },
                        deps.len(),
                        child.cu_graph(),
                    ),
                    "M3 child add2",
                )?;
            }
            prev2 = Some(node);
            children2.push(child);
        }
        let mut exec2: sys::CUgraphExec = std::ptr::null_mut();
        unsafe {
            let _main = e.gpu.enter_main()?;
            cu_try(
                sys::cuGraphInstantiateWithFlags(&mut exec2, parent2, 0),
                "M3 instantiate2",
            )?;
        }
        for _ in 0..3 {
            unsafe {
                let _main = e.gpu.enter_main()?;
                cu_try(
                    sys::cuGraphLaunch(exec2, e.stream().cu_stream() as sys::CUstream),
                    "M3 launch2",
                )?;
            }
        }
        {
            let _main = e.gpu.enter_main()?;
            e.stream().synchronize()?;
        }
        let started = std::time::Instant::now();
        for _ in 0..reps {
            unsafe {
                let _main = e.gpu.enter_main()?;
                cu_try(
                    sys::cuGraphLaunch(exec2, e.stream().cu_stream() as sys::CUstream),
                    "M3 launch2 timed",
                )?;
            }
        }
        {
            let _main = e.gpu.enter_main()?;
            e.stream().synchronize()?;
        }
        let wall2 = started.elapsed().as_secs_f64();
        let per_node_us = wall2 / reps as f64 / (2.0 * k_pairs as f64) * 1e6;
        eprintln!(
            "[tp-graph-probe] M3 control: same-device chain per-node {:.1}us \
             (cross-device join premium = {:.1}us)",
            per_node_us,
            per_edge_us - per_node_us
        );

        // M3b: e <-> rank0 alternating (same DEVICE, the e-glue seam of the token graph).
        // Also report whether the two engines share a CUDA context.
        let ctx_of = |engine: &Engine| -> Result<u64, Box<dyn std::error::Error>> {
            let _main = engine.gpu.enter_main()?;
            let mut ctx: sys::CUcontext = std::ptr::null_mut();
            cu_try(unsafe { sys::cuCtxGetCurrent(&mut ctx) }, "ctx query")?;
            Ok(ctx as u64)
        };
        let (ctx_e, ctx_r0, ctx_r1) = (ctx_of(&e)?, ctx_of(rank0)?, ctx_of(rank1)?);
        eprintln!(
            "[tp-graph-probe] contexts: e={ctx_e:#x} rank0={ctx_r0:#x} rank1={ctx_r1:#x} \
             e_eq_rank0={}",
            ctx_e == ctx_r0
        );
        let mut parent3: sys::CUgraph = std::ptr::null_mut();
        unsafe {
            cu_try(sys::cuGraphCreate(&mut parent3, 0), "M3b cuGraphCreate")?;
        }
        let mut prev3: Option<sys::CUgraphNode> = None;
        let mut children3 = Vec::new();
        for i in 0..(2 * k_pairs) {
            let (child, _r) = if i % 2 == 0 {
                let _main = e.gpu.enter_main()?;
                e.capture_graph_retained_nowarm(|_| {
                    e.scale_inplace(&mut bufs.input_e, 1.0000001, 64)?;
                    Ok(())
                })?
            } else {
                let _main = rank0.gpu.enter_main()?;
                rank0.capture_graph_retained_nowarm(|_| {
                    rank0.scale_inplace(&mut bufs.in0, 1.0000001, 64)?;
                    Ok(())
                })?
            };
            let mut node: sys::CUgraphNode = std::ptr::null_mut();
            let deps = prev3.map(|p| vec![p]).unwrap_or_default();
            unsafe {
                cu_try(
                    sys::cuGraphAddChildGraphNode(
                        &mut node,
                        parent3,
                        if deps.is_empty() {
                            std::ptr::null()
                        } else {
                            deps.as_ptr()
                        },
                        deps.len(),
                        child.cu_graph(),
                    ),
                    "M3b child add",
                )?;
            }
            prev3 = Some(node);
            children3.push(child);
        }
        let mut exec3: sys::CUgraphExec = std::ptr::null_mut();
        unsafe {
            let _main = e.gpu.enter_main()?;
            cu_try(
                sys::cuGraphInstantiateWithFlags(&mut exec3, parent3, 0),
                "M3b instantiate",
            )?;
        }
        for _ in 0..3 {
            unsafe {
                let _main = e.gpu.enter_main()?;
                cu_try(
                    sys::cuGraphLaunch(exec3, e.stream().cu_stream() as sys::CUstream),
                    "M3b launch",
                )?;
            }
        }
        {
            let _main = e.gpu.enter_main()?;
            e.stream().synchronize()?;
        }
        let started = std::time::Instant::now();
        for _ in 0..reps {
            unsafe {
                let _main = e.gpu.enter_main()?;
                cu_try(
                    sys::cuGraphLaunch(exec3, e.stream().cu_stream() as sys::CUstream),
                    "M3b launch timed",
                )?;
            }
        }
        {
            let _main = e.gpu.enter_main()?;
            e.stream().synchronize()?;
        }
        let wall3 = started.elapsed().as_secs_f64();
        eprintln!(
            "[tp-graph-probe] M3b e<->rank0 (same device): per-node {:.1}us",
            wall3 / reps as f64 / (2.0 * k_pairs as f64) * 1e6
        );
    }

    // M4: HOST DISPATCH COST — cudarc launch_builder vs raw cuLaunchKernel with a prebuilt
    // param array (the eager wall's ~4ms/token is user-space dispatch across ~1350 launches;
    // this isolates how much of the per-launch cost the baked-args path can reclaim).
    {
        use cudarc::driver::DevicePtr;
        use cudarc::driver::sys;
        let n = 10_000usize;
        let _main = e.gpu.enter_main()?;
        // Warm the kernel + measure builder path (issue-only: no syncs inside the loop).
        e.scale_inplace(&mut bufs.input_e, 1.0000001, 64)?;
        e.stream().synchronize()?;
        let started = std::time::Instant::now();
        for _ in 0..n {
            e.scale_inplace(&mut bufs.input_e, 1.0000001, 64)?;
        }
        let issue_builder = started.elapsed().as_secs_f64();
        e.stream().synchronize()?;
        // Raw path: same kernel via the raw-module twin, params prebuilt once.
        let raw_f = e.raw_kernel_function("scale_f32")?;
        let stream = e.stream();
        let (ptr, _g) = bufs.input_e.device_ptr(&stream);
        let mut dptr = ptr;
        let mut scale = 1.0000001f32;
        let mut nn = 64i32;
        let mut params = [
            &mut dptr as *mut _ as *mut std::ffi::c_void,
            &mut scale as *mut _ as *mut std::ffi::c_void,
            &mut nn as *mut _ as *mut std::ffi::c_void,
        ];
        let cu_stream = e.stream().cu_stream() as sys::CUstream;
        unsafe {
            let r = sys::cuLaunchKernel(
                raw_f,
                1,
                1,
                1,
                64,
                1,
                1,
                0,
                cu_stream,
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            if r != sys::CUresult::CUDA_SUCCESS {
                return Err(format!("M4 raw warm launch: {r:?}").into());
            }
        }
        e.stream().synchronize()?;
        let started = std::time::Instant::now();
        for _ in 0..n {
            unsafe {
                let r = sys::cuLaunchKernel(
                    raw_f,
                    1,
                    1,
                    1,
                    64,
                    1,
                    1,
                    0,
                    cu_stream,
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                );
                if r != sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("M4 raw launch: {r:?}").into());
                }
            }
        }
        let issue_raw = started.elapsed().as_secs_f64();
        e.stream().synchronize()?;
        eprintln!(
            "[tp-graph-probe] M4 dispatch: builder {:.2}us/launch vs raw-prebuilt {:.2}us/launch              (delta {:.2}us x ~1350 launches/token = {:.2}ms/token reclaimable)",
            issue_builder / n as f64 * 1e6,
            issue_raw / n as f64 * 1e6,
            (issue_builder - issue_raw) / n as f64 * 1e6,
            (issue_builder - issue_raw) / n as f64 * 1350.0 * 1e3,
        );
    }
    Ok(())
}
