//! Full-layer capture instrument. Replay is gate-only and requires the caller to
//! own the stable workspace/scalar sources captured by the graph.
use cudarc::driver::{CudaGraph, CudaStream, sys};
use std::{collections::BTreeMap, sync::Arc};

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use std::ffi::c_void;

/// Capture-only owner. This deliberately does not use `capture_layer`, which
/// executes immediately and drains before its paired rank can launch.
struct ReplayGraph {
    graph: *mut c_void,
    executable: *mut c_void,
    stream: Arc<CudaStream>,
    census: [u64; 7],
}
// The request and TP walk mutex serialize every launch. Handles are immutable
// after instantiate; each operation explicitly binds its owning CUDA context.
unsafe impl Send for ReplayGraph {}
unsafe impl Sync for ReplayGraph {}
impl ReplayGraph {
    fn begin(stream: Arc<CudaStream>) -> Result<Self, String> {
        if stream.context().is_event_tracking() {
            return Err("full replay requires tracking disabled before allocation".into());
        }
        let mut result = Self {
            graph: std::ptr::null_mut(),
            executable: std::ptr::null_mut(),
            stream,
            census: [0; 7],
        };
        result
            .stream
            .context()
            .bind_to_thread()
            .map_err(|e| e.to_string())?;
        unsafe {
            crate::dsv4_ffi::ck(
                "replay capture begin",
                crate::dsv4_ffi::memra_dsv4_replay_capture_begin(
                    &mut result.graph,
                    result.stream.cu_stream().cast(),
                ),
            )?;
        }
        Ok(result)
    }
    fn end(&mut self) -> Result<(), String> {
        self.stream
            .context()
            .bind_to_thread()
            .map_err(|e| e.to_string())?;
        unsafe {
            crate::dsv4_ffi::ck(
                "replay capture end",
                crate::dsv4_ffi::memra_dsv4_replay_capture_end(
                    self.graph,
                    &mut self.executable,
                    self.stream.cu_stream().cast(),
                ),
            )?;
            crate::dsv4_ffi::ck(
                "replay census",
                crate::dsv4_ffi::memra_dsv4_replay_census(self.graph, self.census.as_mut_ptr()),
            )
        }
    }
    fn launch(&self) -> Result<(), String> {
        if self.executable.is_null() {
            return Err("uninstantiated replay graph".into());
        }
        self.stream
            .context()
            .bind_to_thread()
            .map_err(|e| e.to_string())?;
        unsafe {
            crate::dsv4_ffi::ck(
                "replay launch",
                crate::dsv4_ffi::memra_dsv4_replay_launch(
                    self.executable,
                    self.stream.cu_stream().cast(),
                ),
            )
        }
    }
}
impl Drop for ReplayGraph {
    fn drop(&mut self) {
        let result = self
            .stream
            .context()
            .bind_to_thread()
            .map_err(|e| e.to_string())
            .and_then(|()| unsafe {
                crate::dsv4_ffi::ck(
                    "replay graph destroy",
                    crate::dsv4_ffi::memra_dsv4_replay_destroy(
                        self.graph,
                        self.executable,
                        self.stream.cu_stream().cast(),
                    ),
                )
            });
        if let Err(error) = result {
            eprintln!("{error}");
        }
    }
}

/// One request owns both ranks, both segments, stable inputs, device counters
/// and the real sampler. Drop drains BOTH ranks before any captured resource
/// is released. MatrixStep declares this before its workspace/cache references.
pub(crate) struct ReplayPair {
    graphs: [[Option<ReplayGraph>; 2]; 2],
    pub sampler: crate::dsv4_sampler::Dsv4DeviceSampler,
    input: [CudaSlice<u64>; 2],
    host: [crate::PinnedHostBuf; 2],
    counters: [CudaSlice<u64>; 2],
    streams: [Arc<CudaStream>; 2],
    pub cfg: crate::dsv4_gpu::Dsv4SampleCfg,
    pub ready: bool,
    pub owner: usize,
    pub captures: [u64; 2],
    pub ar_blocks: [i32; 2],
}
impl ReplayPair {
    pub fn new(
        streams: [Arc<CudaStream>; 2],
        sampler: crate::dsv4_sampler::Dsv4DeviceSampler,
        cfg: crate::dsv4_gpu::Dsv4SampleCfg,
        owner: usize,
    ) -> Result<Self, String> {
        let mut inputs = Vec::new();
        let mut hosts = Vec::new();
        let mut counts = Vec::new();
        for stream in &streams {
            stream
                .context()
                .bind_to_thread()
                .map_err(|e| e.to_string())?;
            inputs.push(stream.alloc_zeros(3).map_err(|e| e.to_string())?);
            counts.push(stream.alloc_zeros(2).map_err(|e| e.to_string())?);
            hosts.push(crate::PinnedHostBuf::new(24).map_err(|e| e.to_string())?);
            stream.synchronize().map_err(|e| e.to_string())?;
        }
        Ok(Self {
            graphs: [[None, None], [None, None]],
            sampler,
            input: inputs.try_into().map_err(|_| "replay input rank count")?,
            host: hosts.try_into().map_err(|_| "replay host rank count")?,
            counters: counts.try_into().map_err(|_| "replay counter rank count")?,
            streams,
            cfg,
            ready: false,
            owner,
            captures: [0; 2],
            ar_blocks: [
                crate::tp_ar::ar_blocks_for(4096),
                crate::tp_ar::ar_blocks_for(6 * 4096),
            ],
        })
    }
    pub fn upload(&mut self, token: u32, pos: usize, fault: u64) -> Result<(), String> {
        let words = [
            u64::from(token) | ((pos as u64) << 32),
            crate::dsv4_gpu::dsv4_pos_uniform(self.cfg.seed, pos + 1).to_bits(),
            fault,
        ];
        for rank in 0..2 {
            let stream = &self.streams[rank];
            stream
                .context()
                .bind_to_thread()
                .map_err(|e| e.to_string())?;
            let src = unsafe {
                std::slice::from_raw_parts_mut(
                    self.host[rank].as_mut_slice().as_mut_ptr().cast::<u64>(),
                    3,
                )
            };
            src.copy_from_slice(&words);
            stream
                .memcpy_htod(&src[..], &mut self.input[rank])
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
    pub fn begin(&mut self, segment: usize) -> Result<(), String> {
        for rank in 0..2 {
            if self.graphs[rank][segment].is_some() {
                return Err("replay recapture refused".into());
            }
            self.graphs[rank][segment] = Some(ReplayGraph::begin(self.streams[rank].clone())?);
        }
        Ok(())
    }
    pub fn end(&mut self, segment: usize) -> Result<(), String> {
        for rank in 0..2 {
            self.graphs[rank][segment]
                .as_mut()
                .ok_or("replay capture missing")?
                .end()?;
            let census = self.graphs[rank][segment].as_ref().expect("capture").census;
            if census[6] != 0
                || (segment == 0 && (census[2] != 86 || census[3] != 1 || census[4] != 86))
            {
                return Err(format!(
                    "incomplete full-token graph rank {rank} segment {segment}: {census:?}"
                ));
            }
        }
        self.captures[segment] += 1;
        Ok(())
    }
    pub fn launch(&self, segment: usize) -> Result<(), String> {
        // Never drain the first rank before submission of the second.
        for rank in 0..2 {
            self.graphs[rank][segment]
                .as_ref()
                .ok_or("replay graph missing")?
                .launch()?;
        }
        Ok(())
    }
    pub fn census(&self) -> [[[u64; 7]; 2]; 2] {
        std::array::from_fn(|r| {
            std::array::from_fn(|s| self.graphs[r][s].as_ref().map_or([0; 7], |g| g.census))
        })
    }
    pub fn dump(&self, directory: &std::path::Path) -> Result<(), String> {
        self.drain_both()?;
        for rank in 0..2 {
            for segment in 0..2 {
                let graph = self.graphs[rank][segment]
                    .as_ref()
                    .ok_or("graph dump before complete capture")?;
                let name = format!("full-token-rank{rank}-segment{segment}.dot");
                let path = directory.join(name);
                let path = std::ffi::CString::new(path.as_os_str().as_encoded_bytes())
                    .map_err(|e| e.to_string())?;
                graph
                    .stream
                    .context()
                    .bind_to_thread()
                    .map_err(|e| e.to_string())?;
                unsafe {
                    crate::dsv4_ffi::ck(
                        "replay graph dump",
                        crate::dsv4_ffi::memra_dsv4_replay_dump(graph.graph, path.as_ptr()),
                    )?;
                }
            }
        }
        Ok(())
    }
    pub fn input_ptr(&self, rank: usize) -> *const u64 {
        self.input[rank].device_ptr(&self.streams[rank]).0 as *const u64
    }
    pub fn counter_ptr(&mut self, rank: usize, segment: usize) -> *mut u64 {
        unsafe {
            (self.counters[rank].device_ptr_mut(&self.streams[rank]).0 as *mut u64).add(segment)
        }
    }
    pub fn counts(&self) -> Result<[[u64; 2]; 2], String> {
        let mut out = [[0; 2]; 2];
        for (rank, row) in out.iter_mut().enumerate() {
            self.streams[rank]
                .context()
                .bind_to_thread()
                .map_err(|e| e.to_string())?;
            self.streams[rank]
                .memcpy_dtoh(&self.counters[rank], row)
                .map_err(|e| e.to_string())?;
            self.streams[rank]
                .synchronize()
                .map_err(|e| e.to_string())?;
        }
        Ok(out)
    }
    pub fn abort_capture_both(&self) -> Result<(), String> {
        let mut errors = Vec::new();
        for (rank, stream) in self.streams.iter().enumerate() {
            let result = stream
                .context()
                .bind_to_thread()
                .map_err(|e| e.to_string())
                .and_then(|()| unsafe {
                    crate::dsv4_ffi::ck(
                        "replay capture abort",
                        crate::dsv4_ffi::memra_dsv4_replay_abort(stream.cu_stream().cast()),
                    )
                });
            if let Err(error) = result {
                errors.push(format!("rank {rank}: {error}"));
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("; "))
        }
    }
    pub fn drain_both(&self) -> Result<(), String> {
        let mut errors = Vec::new();
        for (rank, stream) in self.streams.iter().enumerate() {
            let result = stream
                .context()
                .bind_to_thread()
                .and_then(|()| stream.synchronize());
            if let Err(error) = result {
                errors.push(format!("rank {rank}: {error}"));
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("; "))
        }
    }
}
impl Drop for ReplayPair {
    fn drop(&mut self) {
        // Abort capture first: synchronizing a captured stream is prohibited.
        for stream in &self.streams {
            if let Err(error) = stream.context().bind_to_thread() {
                eprintln!("replay drop bind: {error}");
                continue;
            }
            if stream
                .capture_status()
                .ok()
                .is_some_and(|s| s != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE)
            {
                // The graph is owned by ReplayGraph, so end capture without
                // instantiation and without destroying its returned graph.
                unsafe {
                    crate::dsv4_ffi::memra_dsv4_replay_abort(stream.cu_stream().cast());
                }
            }
        }
        if let Err(error) = self.drain_both() {
            eprintln!("replay pair drop: {error}");
        }
    }
}

pub struct Dsv4LayerCapture {
    pub layer: usize,
    pub nodes: BTreeMap<String, usize>,
    pub kernels: Vec<String>,
    graph: CudaGraph,
}
impl std::fmt::Debug for Dsv4LayerCapture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dsv4LayerCapture")
            .field("layer", &self.layer)
            .field("nodes", &self.nodes)
            .field("kernels", &self.kernels)
            .finish_non_exhaustive()
    }
}

impl Dsv4LayerCapture {
    /// Replay the retained graph on its capture stream. This is intentionally a
    /// gate-only API: callers must keep the captured workspace and scalar-source
    /// buffers alive for the graph lifetime.
    pub(crate) fn replay(&self) -> Result<(), String> {
        self.graph
            .launch()
            .map_err(|e| format!("layer {} graph replay: {e}", self.layer))
    }
}

struct CaptureScope {
    stream: Arc<CudaStream>,
    begun: bool,
}

impl Drop for CaptureScope {
    fn drop(&mut self) {
        if self.begun {
            // A failed/panicking body must not leave the owning stream capturing.
            let _ = self.stream.end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            );
        }
    }
}

pub(crate) fn capture_layer(
    stream: Arc<CudaStream>,
    layer: usize,
    body: impl FnOnce() -> Result<(), String>,
) -> Result<Dsv4LayerCapture, String> {
    // Cudarc's switch prevents creation of new per-buffer events, but existing
    // buffers retain theirs and their pointer guards still record those events.
    // Disabling here is too late. DSV4 disables tracking before loading tensors; refuse
    // the tracking-on mode rather than silently making an unsafe transition.
    if stream.context().is_event_tracking() {
        return Err(format!(
            "layer {layer} capture requires event tracking disabled before buffer allocation"
        ));
    }
    let fail = |step: &str, error: &dyn std::fmt::Display| {
        format!("layer {layer} capture {step}: {error}")
    };
    if stream.capture_status().map_err(|e| fail("status", &e))?
        != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
    {
        return Err(format!(
            "layer {layer} capture requires an uncaptured stream"
        ));
    }
    stream.synchronize().map_err(|e| fail("pre-drain", &e))?;
    let mut scope = CaptureScope {
        stream: stream.clone(),
        begun: false,
    };
    stream
        .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        .map_err(|e| fail("begin", &e))?;
    scope.begun = true;
    // No warmup: capture records the body but does not execute cache mutations.
    let result = body();
    scope.begun = false;
    let ended = stream.end_capture(
        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    );
    result.map_err(|e| fail("body", &e))?;
    let graph = ended
        .map_err(|e| fail("end/instantiate", &e))?
        .ok_or_else(|| format!("layer {layer} capture produced no graph"))?;
    let nodes = crate::graph_update::node_census(&graph).map_err(|e| fail("node census", &e))?;
    let kernels = crate::graph_update::kernel_nodes(&graph)
        .map_err(|e| fail("kernel census", &e))?
        .into_iter()
        .map(|node| node.name)
        .collect();
    graph.upload().map_err(|e| fail("upload", &e))?;
    graph.launch().map_err(|e| fail("launch", &e))?;
    stream.synchronize().map_err(|e| fail("completion", &e))?;
    Ok(Dsv4LayerCapture {
        layer,
        nodes,
        kernels,
        graph,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};

    #[test]
    #[ignore = "requires an exclusively locked non-serving CUDA device"]
    fn cuda_capture_runs_once_and_restores_scope_after_failure() {
        let context = CudaContext::new(0).expect("context");
        let stream = context.new_stream().expect("stream");
        let refusal = capture_layer(stream.clone(), 0, || {
            panic!("tracking-on body must not run")
        });
        assert!(refusal.unwrap_err().contains("before buffer allocation"));
        assert!(
            context.is_event_tracking(),
            "refusal must not change context policy"
        );
        // Same policy and ordering as Dsv4Gpu::load: disable before
        // any buffers are allocated, not around the capture itself.
        unsafe { context.disable_event_tracking() };
        let values: Vec<f32> = (0..64).map(|x| x as f32 + 1.0).collect();
        let input = stream.clone_htod(&values).expect("input");
        let mut output = stream.alloc_zeros::<f32>(64).expect("output");
        let tracking = context.is_event_tracking();
        let body_calls = std::cell::Cell::new(0);
        let captured = capture_layer(stream.clone(), 0, || {
            body_calls.set(body_calls.get() + 1);
            unsafe {
                crate::dsv4_ffi::ck(
                    "capture add",
                    crate::dsv4_ffi::memra_dsv4_add_inplace(
                        output.device_ptr_mut(&stream).0 as *mut f32,
                        input.device_ptr(&stream).0 as *const f32,
                        64,
                        stream.cu_stream().cast(),
                    ),
                )
            }
        })
        .expect("capture and execute");
        assert_eq!(
            body_calls.get(),
            1,
            "capture must not warm up stateful operations"
        );
        assert_eq!(
            stream.clone_dtoh(&output).unwrap(),
            values,
            "one GPU execution, not warmup plus replay"
        );
        assert_eq!(captured.kernels.len(), 1);
        assert!(captured.kernels[0].contains("dsv4_add_inplace"));
        assert_eq!(context.is_event_tracking(), tracking);

        let failure = capture_layer(stream.clone(), 1, || Err("deliberate body failure".into()));
        let failure = failure.unwrap_err();
        assert!(
            failure.contains("deliberate body failure"),
            "unexpected capture failure: {failure}"
        );
        assert_eq!(
            stream.capture_status().unwrap(),
            sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
        );
        assert_eq!(context.is_event_tracking(), tracking);
        let recovered = capture_layer(stream.clone(), 2, || {
            stream
                .memcpy_dtod(&input, &mut output)
                .map_err(|e| e.to_string())
        })
        .expect("capture after failure");
        assert!(!recovered.nodes.is_empty());
        assert_eq!(stream.clone_dtoh(&output).unwrap(), values);
        println!(
            "PASS layer capture scope: tracking-on refusal, one execution, kernel census, body-error cleanup and subsequent capture"
        );
    }
}
