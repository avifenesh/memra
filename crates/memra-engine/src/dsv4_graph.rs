//! Full-layer capture instrument. Replay is gate-only and requires the caller to
//! own the stable workspace/scalar sources captured by the graph.
use cudarc::driver::{CudaGraph, CudaStream, sys};
use std::{collections::BTreeMap, sync::Arc};

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use std::ffi::c_void;

/// An unsuccessful stream drain is not a completion receipt. Never unwind or
/// return through owners of peer-visible storage when either rank is unproven.
/// Try both ranks, print every failure, then fail-stop without running Drop.
pub(crate) fn require_pair_completion(
    context: &str,
    mut drain: impl FnMut(usize) -> Result<(), String>,
) {
    let mut errors = Vec::new();
    for rank in 0..2 {
        if let Err(error) = drain(rank) {
            errors.push(format!("rank {rank}: {error}"));
        }
    }
    if !errors.is_empty() {
        replay_fail_stop(context, &errors.join("; "));
    }
}

fn replay_fail_stop(context: &str, error: &str) -> ! {
    use std::io::Write;
    let _ = writeln!(
        std::io::stderr(),
        "FATAL full-token replay completion unproven ({context}): {error}; aborting before captured storage release"
    );
    let _ = std::io::stderr().flush();
    std::process::abort();
}

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
    // Slots 0/1 retain the original full-forward/commit ABI. Cadence mode uses
    // forward slots 0 (ordinary), 2 (C4), 3 (C4+C128), sharing commit slot 1.
    graphs: [[Option<ReplayGraph>; 4]; 2],
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
    pub cadence: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReplayCadence {
    Ordinary,
    C4,
    C128,
}
impl ReplayCadence {
    pub fn for_position(pos: usize) -> Self {
        if (pos + 1).is_multiple_of(128) {
            Self::C128
        } else if (pos + 1).is_multiple_of(4) {
            Self::C4
        } else {
            Self::Ordinary
        }
    }
    pub fn slot(self) -> usize {
        match self {
            Self::Ordinary => 0,
            Self::C4 => 2,
            Self::C128 => 3,
        }
    }
    pub fn emits(self, ratio: usize) -> bool {
        match ratio {
            4 => self != Self::Ordinary,
            128 => self == Self::C128,
            _ => false,
        }
    }
}
impl ReplayPair {
    pub fn new(
        streams: [Arc<CudaStream>; 2],
        sampler: crate::dsv4_sampler::Dsv4DeviceSampler,
        cfg: crate::dsv4_gpu::Dsv4SampleCfg,
        owner: usize,
        cadence: bool,
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
            counts.push(stream.alloc_zeros(4).map_err(|e| e.to_string())?);
            hosts.push(crate::PinnedHostBuf::new(24).map_err(|e| e.to_string())?);
            stream.synchronize().map_err(|e| e.to_string())?;
        }
        Ok(Self {
            graphs: std::array::from_fn(|_| std::array::from_fn(|_| None)),
            sampler,
            input: inputs.try_into().map_err(|_| "replay input rank count")?,
            host: hosts.try_into().map_err(|_| "replay host rank count")?,
            counters: counts.try_into().map_err(|_| "replay counter rank count")?,
            streams,
            cfg,
            ready: false,
            owner,
            cadence,
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
        if segment >= 4 || (!self.cadence && segment >= 2) {
            return Err("unsupported replay graph slot".into());
        }
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
                || (segment != 1 && (census[2] != 86 || census[3] != 1 || census[4] != 86))
            {
                return Err(format!(
                    "incomplete full-token graph rank {rank} segment {segment}: {census:?}"
                ));
            }
        }
        self.captures[usize::from(segment == 1)] += 1;
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
    pub fn variant_census(&self) -> [[[u64; 7]; 4]; 2] {
        std::array::from_fn(|r| {
            std::array::from_fn(|s| self.graphs[r][s].as_ref().map_or([0; 7], |g| g.census))
        })
    }
    pub fn dump(&self, directory: &std::path::Path) -> Result<(), String> {
        self.drain_both()?;
        for rank in 0..2 {
            for segment in 0..if self.cadence { 4 } else { 2 } {
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
        Ok(self.variant_counts()?.map(|r| [r[0] + r[2] + r[3], r[1]]))
    }
    pub fn variant_counts(&self) -> Result<[[u64; 4]; 2], String> {
        let mut out = [[0; 4]; 2];
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
        require_pair_completion("ReplayPair drain", |rank| {
            let stream = &self.streams[rank];
            stream
                .context()
                .bind_to_thread()
                .and_then(|()| stream.synchronize())
                .map_err(|e| e.to_string())
        });
        Ok(())
    }
}
impl Drop for ReplayPair {
    fn drop(&mut self) {
        // Abort capture first: synchronizing a captured stream is prohibited.
        if let Err(error) = self.abort_capture_both() {
            replay_fail_stop("ReplayPair drop capture abort", &error);
        }
        if let Err(error) = self.drain_both() {
            replay_fail_stop("ReplayPair drop drain", &error);
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
    fn replay_completion_policy_fail_stop_subprocess() {
        const CHILD: &str = "MEMRA_TEST_REPLAY_DRAIN_FAILURE_CHILD";
        if let Ok(value) = std::env::var(CHILD) {
            struct CapturedStorage;
            impl Drop for CapturedStorage {
                fn drop(&mut self) {
                    eprintln!("CAPTURED_STORAGE_RELEASED");
                }
            }
            let _request_workspace = CapturedStorage;
            let (scenario, rank) = value.split_once(':').unwrap();
            let failed_rank = rank.parse::<usize>().unwrap();
            let drain = |rank| {
                eprintln!("DRAIN_ATTEMPTED_{rank}");
                if rank == failed_rank {
                    Err("injected drain failure".into())
                } else {
                    Ok(())
                }
            };
            if scenario == "drop" {
                struct CompletionGuard<F: FnMut(usize) -> Result<(), String>>(F);
                impl<F: FnMut(usize) -> Result<(), String>> Drop for CompletionGuard<F> {
                    fn drop(&mut self) {
                        require_pair_completion("destructor unwind", &mut self.0);
                    }
                }
                let _guard = CompletionGuard(drain);
                panic!("original submission exception");
            }
            require_pair_completion("injected failed CUDA completion", drain);
            panic!("failed completion returned instead of aborting");
        }
        for scenario in ["error", "drop"] {
            for rank in 0..2 {
                use std::os::unix::process::ExitStatusExt;
                let output = std::process::Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--exact",
                        "dsv4_graph::tests::replay_completion_policy_fail_stop_subprocess",
                        "--nocapture",
                    ])
                    .env(CHILD, format!("{scenario}:{rank}"))
                    .output()
                    .unwrap();
                let stderr = String::from_utf8_lossy(&output.stderr);
                assert_eq!(output.status.signal(), Some(libc::SIGABRT), "{stderr}");
                assert!(
                    stderr.contains("DRAIN_ATTEMPTED_0") && stderr.contains("DRAIN_ATTEMPTED_1"),
                    "{stderr}"
                );
                assert!(
                    stderr.contains("injected drain failure")
                        && stderr.contains("completion unproven"),
                    "{stderr}"
                );
                assert!(
                    !stderr.contains("CAPTURED_STORAGE_RELEASED"),
                    "destructor ran: {stderr}"
                );
            }
        }
        let mut attempts = Vec::new();
        require_pair_completion("successful drains after ordinary refusal", |rank| {
            attempts.push(rank);
            Ok(())
        });
        assert_eq!(attempts, [0, 1]);
    }

    #[test]
    #[ignore = "requires the exclusively locked development pair; runtime lifetime gate, not a model gate"]
    fn replay_rust_partial_submission_and_capture_cleanup() {
        let contexts = [CudaContext::new(0).unwrap(), CudaContext::new(1).unwrap()];
        for ctx in &contexts {
            unsafe {
                ctx.disable_event_tracking();
            }
        }
        let streams = [
            contexts[0].new_stream().unwrap(),
            contexts[1].new_stream().unwrap(),
        ];
        let make_pair = |cadence| {
            let sampler =
                crate::dsv4_sampler::Dsv4DeviceSampler::new(streams[1].clone(), 257).unwrap();
            ReplayPair::new(
                streams.clone(),
                sampler,
                crate::dsv4_gpu::Dsv4SampleCfg {
                    temperature: 1.0,
                    top_p: 1.0,
                    top_k: 0,
                    seed: 1,
                },
                0,
                cadence,
            )
            .unwrap()
        };
        let assert_uncaptured = || {
            for stream in &streams {
                stream.context().bind_to_thread().unwrap();
                assert_eq!(
                    stream.capture_status().unwrap(),
                    sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
                );
                stream.synchronize().unwrap();
            }
        };
        // Actual Rust owner unwinding from a capture-body failure. No execution
        // occurred; Drop must end BOTH capture scopes before its paired drain.
        let capture_error = (|| -> Result<(), String> {
            let mut pair = make_pair(false);
            pair.begin(0)?;
            Err("injected capture body failure".into())
        })()
        .unwrap_err();
        assert_eq!(capture_error, "injected capture body failure");
        assert_uncaptured();
        // End/instantiate error at the real Rust capture_end FFI boundary.
        {
            let mut pair = make_pair(false);
            pair.begin(0).unwrap();
            pair.abort_capture_both().unwrap();
            let error = pair.graphs[0][0].as_mut().unwrap().end().unwrap_err();
            assert!(error.contains("replay capture end"), "{error}");
            // This deliberately calls EndCapture on a stream already ended by
            // abort_capture_both. Consume only its asserted CUDA API last error
            // before testing later launches; do not mask an unexpected failure.
            assert!(error.ends_with("rc=10401"), "unexpected end error: {error}");
            unsafe extern "C" {
                fn cudaGetLastError() -> i32;
            }
            assert_eq!(
                unsafe { cudaGetLastError() },
                401,
                "expected illegal-state error"
            );
            eprintln!("PASS Rust capture-end failure rc=10401 asserted and consumed");
        }
        assert_uncaptured();
        // Use the runtime's actual pair-launch loop and real graph counters.
        // These are tiny no-peer graphs; they cover Rust ownership/submission,
        // not the C++ fixture's peer barriers or full-model layer coverage.
        for (cadence, segment) in [
            (false, 0),
            (false, 1),
            (true, 0),
            (true, 1),
            (true, 2),
            (true, 3),
        ] {
            let mut pair = make_pair(cadence);
            pair.begin(segment).unwrap();
            for (rank, stream) in streams.iter().enumerate() {
                stream.context().bind_to_thread().unwrap();
                unsafe {
                    crate::dsv4_ffi::ck(
                        "test replay tick",
                        crate::dsv4_ffi::memra_dsv4_replay_tick(
                            pair.counter_ptr(rank, segment),
                            stream.cu_stream().cast(),
                        ),
                    )
                    .unwrap();
                }
            }
            // Finish primitive graphs directly: the full-forward census gate
            // deliberately rejects tiny fixtures. Production census is untouched.
            for rank in 0..2 {
                pair.graphs[rank][segment].as_mut().unwrap().end().unwrap();
            }
            let peer = pair.graphs[1][segment].take();
            let error = pair.launch(segment).unwrap_err();
            assert!(error.contains("replay graph missing"), "{error}");
            pair.drain_both().unwrap();
            let counts = pair.variant_counts().unwrap();
            assert_eq!(counts[0][segment], 1, "first rank never submitted");
            assert_eq!(counts[1][segment], 0, "second rank unexpectedly submitted");
            pair.graphs[1][segment] = peer;
            drop(pair);
            assert_uncaptured();
            eprintln!(
                "PASS Rust partial segment={segment} first_submit=1 second_submit=0 paired_completion=1"
            );
        }
        // Exercise the actual retained variant owner and live input upload with
        // all three forward slots, decreasing positions and ring-wrap boundaries.
        let mut live: [CudaSlice<i32>; 2] =
            std::array::from_fn(|rank| streams[rank].alloc_zeros(3).unwrap());
        for stream in &streams {
            stream.synchronize().unwrap();
        }
        // Locals drop in reverse order: the pair must drain/abort before live.
        let mut pair = make_pair(true);
        for slot in [0, 2, 3, 1] {
            pair.begin(slot).unwrap();
            for rank in 0..2 {
                let stream = &streams[rank];
                stream.context().bind_to_thread().unwrap();
                unsafe {
                    let fields = live[rank].device_ptr_mut(stream).0 as *mut i32;
                    let rc = if slot == 1 {
                        crate::dsv4_ffi::memra_dsv4_replay_tick(
                            pair.counter_ptr(rank, slot),
                            stream.cu_stream().cast(),
                        )
                    } else {
                        crate::dsv4_ffi::memra_dsv4_replay_input(
                            pair.input_ptr(rank),
                            fields,
                            fields.add(1),
                            fields.add(2),
                            128,
                            pair.counter_ptr(rank, slot),
                            stream.cu_stream().cast(),
                        )
                    };
                    crate::dsv4_ffi::ck("cadence input fixture", rc).unwrap();
                }
            }
            for rank in 0..2 {
                pair.graphs[rank][slot].as_mut().unwrap().end().unwrap();
            }
        }
        assert_eq!(
            pair.variant_counts().unwrap(),
            [[0; 4]; 2],
            "capture executed controls"
        );
        let mut expected = [0u64; 4];
        for (index, (pos, slot)) in [
            (0, 0),
            (3, 2),
            (127, 3),
            (128, 0),
            (255, 3),
            (256, 0),
            (383, 3),
            (511, 3),
            (300, 0),
            (3, 2),
            (4, 0),
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(ReplayCadence::for_position(pos).slot(), slot);
            let token = (index as u32 + 1) * 17;
            pair.upload(token, pos, 0).unwrap();
            pair.launch(slot).unwrap();
            pair.launch(1).unwrap();
            pair.drain_both().unwrap();
            expected[slot] += 1;
            expected[1] += 1;
            assert_eq!(pair.variant_counts().unwrap(), [expected; 2]);
            for rank in 0..2 {
                let mut fields = [0i32; 3];
                let mut words = [0u64; 3];
                streams[rank].memcpy_dtoh(&live[rank], &mut fields).unwrap();
                streams[rank]
                    .memcpy_dtoh(&pair.input[rank], &mut words)
                    .unwrap();
                streams[rank].synchronize().unwrap();
                assert_eq!(fields, [token as i32, pos as i32, (pos % 128) as i32]);
                assert_eq!(
                    words,
                    [
                        u64::from(token) | ((pos as u64) << 32),
                        crate::dsv4_gpu::dsv4_pos_uniform(1, pos + 1).to_bits(),
                        0
                    ]
                );
            }
        }
        drop(pair);
        assert_uncaptured();
        eprintln!(
            "PASS Rust cadence slots=4 changing_tokens=11 decreasing_positions=1 per_variant_counters=1 full_uniform_bits=1 model_layers=0"
        );
    }

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
