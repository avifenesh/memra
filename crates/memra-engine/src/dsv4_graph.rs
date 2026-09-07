//! CUDA graph capture primitives. Full-layer capture remains diagnostic; the
//! retained Graph-B segment is gate-only and requires the caller to own the
//! stable workspace/scalar sources captured by the graph.
use cudarc::driver::{CudaGraph, CudaStream, sys};
use std::{collections::BTreeMap, sync::Arc};

/// The only persistent graph segment currently owned by the DSV4 verifier.
/// The executable is deliberately narrower than a layer: cache/index/C4/EP/PP
/// work stays eager and the segment begins after the eager C4 gather.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum Dsv4GraphSegment {
    GraphB,
}

/// Runtime identity of a captured Graph-B body.
///
/// Dynamic scalar *contents* (position and block counts) are intentionally not
/// part of this key: they are read from the stable device scalar buffers during
/// replay.  Every pointer that the captured body dereferences is part of the
/// identity, as are the realized slot/shape and math arms.  A changed pointer
/// therefore cannot accidentally replay an executable that still targets the
/// old workspace.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct Dsv4GraphKey {
    pub(crate) segment: Dsv4GraphSegment,
    pub(crate) layer: usize,
    pub(crate) stage: usize,
    pub(crate) tokens: usize,
    pub(crate) attention_topk: usize,
    pub(crate) slots: usize,
    pub(crate) idx_stride: usize,
    pub(crate) ratio: usize,
    pub(crate) arm: u32,
    pub(crate) input_h: usize,
    pub(crate) pos_dev: usize,
    pub(crate) q: usize,
    pub(crate) attention_kv: usize,
    pub(crate) attention_indices: usize,
    pub(crate) sink: usize,
    pub(crate) sink_scores: usize,
    pub(crate) sink_evals: usize,
    pub(crate) sink_den: usize,
    pub(crate) o: usize,
    pub(crate) o_b: usize,
    pub(crate) og: usize,
    pub(crate) attn_out: usize,
    pub(crate) h_b: usize,
    pub(crate) post: usize,
    pub(crate) comb: usize,
    pub(crate) gemm_xb: usize,
    pub(crate) y_hc: usize,
    pub(crate) xf: usize,
    pub(crate) ffn_norm: usize,
    pub(crate) hc_ffn_fn: usize,
    pub(crate) hc_ffn_base_dev: usize,
    pub(crate) hc_ffn_scale_dev: usize,
    pub(crate) fc: usize,
    pub(crate) layer_weights: usize,
}

impl Dsv4GraphKey {
    pub(crate) fn graph_b_valid(&self) -> bool {
        self.segment == Dsv4GraphSegment::GraphB
            && self.tokens == 1
            && self.slots > 0
            && self.idx_stride >= self.slots
            && self.input_h != 0
            && self.pos_dev != 0
            && self.attention_kv != 0
            && self.attention_indices != 0
            && self.q != 0
            && self.o != 0
            && self.h_b != 0
            && self.post != 0
            && self.comb != 0
            && self.gemm_xb != 0
            && self.xf != 0
            && self.ffn_norm != 0
            && self.hc_ffn_fn != 0
            && self.hc_ffn_base_dev != 0
            && self.hc_ffn_scale_dev != 0
            && self.fc != 0
            && self.layer_weights != 0
    }

    pub(crate) fn pointer_identity_changed(&self, other: &Self) -> bool {
        self.input_h != other.input_h
            || self.pos_dev != other.pos_dev
            || self.q != other.q
            || self.attention_kv != other.attention_kv
            || self.attention_indices != other.attention_indices
            || self.sink != other.sink
            || self.sink_scores != other.sink_scores
            || self.sink_evals != other.sink_evals
            || self.sink_den != other.sink_den
            || self.o != other.o
            || self.o_b != other.o_b
            || self.og != other.og
            || self.attn_out != other.attn_out
            || self.h_b != other.h_b
            || self.post != other.post
            || self.comb != other.comb
            || self.gemm_xb != other.gemm_xb
            || self.y_hc != other.y_hc
            || self.xf != other.xf
            || self.ffn_norm != other.ffn_norm
            || self.hc_ffn_fn != other.hc_ffn_fn
            || self.hc_ffn_base_dev != other.hc_ffn_base_dev
            || self.hc_ffn_scale_dev != other.hc_ffn_scale_dev
            || self.fc != other.fc
            || self.layer_weights != other.layer_weights
    }
}

/// Capture metadata used by the Graph-B gate.  Keeping this separate from the
/// executable handle makes the stats API serializable without exposing CUDA
/// graph internals.
#[derive(Clone, Debug, Default)]
pub(crate) struct Dsv4GraphStats {
    pub(crate) captures: u64,
    pub(crate) replays: u64,
    pub(crate) invalidations: u64,
    pub(crate) nodes: u64,
    pub(crate) kernels: u64,
    pub(crate) wo_a_replay_nodes: u64,
}

pub(crate) fn graph_b_wo_a_kernel_nodes(capture: &Dsv4LayerCapture) -> usize {
    capture
        .kernels
        .iter()
        .filter(|name| is_graph_b_grouped_wo_a_kernel(name))
        .count()
}

/// The Graph-B census stores the CUDA driver's raw symbol name.  The grouped
/// wo_a path is the m=1, grouped=true instantiation of the regular FP8 batched
/// GEMV template.  Do not match the Rust FFI wrapper name: the wrapper is not a
/// graph kernel node, and the regular `<1, false>` wo_b GEMV must not count as
/// replayed wo_a work.
pub(crate) fn is_graph_b_grouped_wo_a_kernel(name: &str) -> bool {
    if !name.contains("dsv4_gemv_fp8_m_kernel") {
        return false;
    }
    name.contains("ILi1ELb1E") || name.contains("<1, true>") || name.contains("<1,true>")
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

/// Capture one persistent segment and execute its first step once.
///
/// This wrapper is intentionally separate from the older diagnostic
/// `capture_layer` name so callers cannot accidentally treat a partial Graph-B
/// capture as a complete model layer.  The body is submitted exactly once by
/// capture and the resulting graph is launched exactly once before returning;
/// subsequent calls must use [`Dsv4LayerCapture::replay`] and must not resubmit
/// the body closure.
pub(crate) fn capture_segment(
    stream: Arc<CudaStream>,
    segment: Dsv4GraphSegment,
    key: Dsv4GraphKey,
    body: impl FnOnce() -> Result<(), String>,
) -> Result<(Dsv4GraphKey, Dsv4LayerCapture), String> {
    if segment != key.segment {
        return Err("graph segment/key mismatch".into());
    }
    if segment == Dsv4GraphSegment::GraphB && !key.graph_b_valid() {
        return Err("invalid Graph-B shape or pointer identity".into());
    }
    let capture = capture_layer(stream, key.layer, body)?;
    Ok((key, capture))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};

    fn graph_b_key(slots: usize, input_h: usize) -> Dsv4GraphKey {
        Dsv4GraphKey {
            segment: Dsv4GraphSegment::GraphB,
            layer: 3,
            stage: 1,
            tokens: 1,
            attention_topk: 512,
            slots,
            idx_stride: 640,
            ratio: 128,
            arm: 0x17,
            input_h,
            pos_dev: 0x2000,
            q: 0x3000,
            attention_kv: 0x4000,
            attention_indices: 0x5000,
            sink: 0x6000,
            sink_scores: 0x7000,
            sink_evals: 0x8000,
            sink_den: 0x9000,
            o: 0xa000,
            o_b: 0xb000,
            og: 0xc000,
            attn_out: 0xd000,
            h_b: 0xe000,
            post: 0xe100,
            comb: 0xe200,
            gemm_xb: 0xe300,
            y_hc: 0xf000,
            xf: 0x10000,
            ffn_norm: 0x11000,
            hc_ffn_fn: 0x11100,
            hc_ffn_base_dev: 0x11200,
            hc_ffn_scale_dev: 0x11300,
            fc: 0x12000,
            layer_weights: 0x13000,
        }
    }

    #[test]
    fn graph_b_key_requires_real_shape_and_pointers() {
        let key = graph_b_key(512, 0x1000);
        assert!(key.graph_b_valid());
        assert_ne!(key, graph_b_key(513, 0x1000));
        assert_ne!(key, graph_b_key(512, 0x1001));
        let mut invalid = key;
        invalid.idx_stride = 511;
        assert!(!invalid.graph_b_valid());
        invalid = key;
        invalid.tokens = 2;
        assert!(!invalid.graph_b_valid());
    }

    #[test]
    fn graph_b_wo_a_census_matches_actual_grouped_template_only() {
        assert!(is_graph_b_grouped_wo_a_kernel(
            "_Z22dsv4_gemv_fp8_m_kernelILi1ELb1EEvPKh"
        ));
        assert!(is_graph_b_grouped_wo_a_kernel(
            "dsv4_gemv_fp8_m_kernel<1, true>"
        ));
        assert!(!is_graph_b_grouped_wo_a_kernel(
            "_Z22dsv4_gemv_fp8_m_kernelILi1ELb0EEvPKh"
        ));
        assert!(!is_graph_b_grouped_wo_a_kernel(
            "_Z22dsv4_gemv_fp8_m_kernelILi2ELb1EEvPKh"
        ));
        assert!(!is_graph_b_grouped_wo_a_kernel(
            "memra_dsv4_gemv_fp8_grouped_m1"
        ));
    }

    #[test]
    fn graph_b_key_leaves_live_scalar_contents_out_of_identity() {
        // Position and block-count contents are device-side scalar updates. They
        // are not represented in Dsv4GraphKey, so a later decode step can replay
        // the same pointer topology without a false cache miss.
        let first = graph_b_key(512, 0x1000);
        let later = graph_b_key(512, 0x1000);
        assert_eq!(first, later);
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
