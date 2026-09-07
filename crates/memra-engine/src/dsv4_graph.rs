//! Full-layer capture instrument. Replay is gate-only and requires the caller to
//! own the stable workspace/scalar sources captured by the graph.
use cudarc::driver::{CudaGraph, CudaStream, sys};
use std::{collections::BTreeMap, sync::Arc};

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
