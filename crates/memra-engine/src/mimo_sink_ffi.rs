//! Bounded MiMo f32 sink-attention decode component.
//! This does not admit MiMo to the serving backend.

use core::ffi::c_void;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use memra_gguf::model_plan::{
    AttentionPlan, AttentionScale, TensorPresence, ValueNorm, ValueProjection,
};

use crate::Engine;

unsafe extern "C" {
    /// Q [64,192], K [seq,kv_heads,192], V [seq,kv_heads,128],
    /// optional sink [64], output [64,128]. All non-null pointers address
    /// contiguous device f32 storage. `seq` includes the current token.
    ///
    /// Full layers require kv_heads=4, window=0, and no sink. Sliding layers
    /// require kv_heads=8, window=128, and a sink. Accepts seq=1..4096.
    /// Returns 0 on accepted asynchronous launch,
    /// 40001..40005 on invalid arguments, or 10000 + cudaError_t for a CUDA
    /// runtime or launch error. Synchronize the stream to catch execution
    /// failures. Callers must keep input and output storage alive until then.
    fn memra_mimo_sink_attn_decode_f32(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        sink: *const f32,
        output: *mut f32,
        seq: i32,
        heads: i32,
        kv_heads: i32,
        qk_dim: i32,
        v_dim: i32,
        window: i32,
        stream: *mut c_void,
    ) -> i32;
}

fn validate_plan(
    attention: &AttentionPlan,
    has_sink: bool,
) -> Result<(usize, usize), &'static str> {
    let (plan, window) = match attention {
        AttentionPlan::Full(plan) => (plan, 0),
        AttentionPlan::SlidingWindow { attention, window } => (attention, *window as usize),
        _ => return Err("MiMo component requires full or sliding attention"),
    };
    let math = plan
        .mimo_math
        .ok_or("MiMo attention plan has no family math")?;
    if math.fused_qkv_checkpoint_shards != Some(4)
        || math.value_scale_before_cache.to_bits() != 0.707f32.to_bits()
        || plan.query_heads != 64
        || plan.key_head_dim != 192
        || plan.value_head_dim != 128
        || plan.qk_norm != TensorPresence::Absent
        || plan.scale != AttentionScale::InverseSqrtKeyDim
        || plan.value_projection != ValueProjection::Separate
        || plan.value_norm != ValueNorm::None
    {
        return Err("MiMo attention plan differs from pinned source geometry");
    }
    match (plan.kv_heads, window, math.sink, has_sink) {
        (4, 0, TensorPresence::Absent, false) => Ok((4, 0)),
        (8, 128, TensorPresence::Required, true) => Ok((8, 128)),
        _ => Err("MiMo attention window, KV heads, and sink do not match"),
    }
}

impl Engine {
    /// Compute one decode query against a bounded, current-token-inclusive
    /// contiguous KV sequence. Query and key must already have RoPE applied;
    /// `value` must already carry the 0.707 scaling from the QKV gather.
    /// This component synchronizes to catch device errors.
    pub fn mimo_sink_decode(
        &self,
        query: &CudaSlice<f32>,
        key: &CudaSlice<f32>,
        value: &CudaSlice<f32>,
        sink: Option<&CudaSlice<f32>>,
        seq: usize,
        attention: &AttentionPlan,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        if !(1..=4096).contains(&seq) {
            return Err("MiMo attention component accepts 1..=4096 tokens".into());
        }
        let (kv_heads, window) = validate_plan(attention, sink.is_some())?;
        if query.len() != 64 * 192
            || key.len() != seq * kv_heads * 192
            || value.len() != seq * kv_heads * 128
            || sink.is_some_and(|slice| slice.len() != 64)
        {
            return Err("MiMo attention input length mismatch".into());
        }
        self.gpu.ctx.bind_to_thread()?;
        let stream = self.stream();
        let device = stream.context().ordinal();
        if query.ordinal() != device
            || key.ordinal() != device
            || value.ordinal() != device
            || sink.is_some_and(|slice| slice.ordinal() != device)
        {
            return Err("MiMo attention input crossed GPU devices".into());
        }
        let mut output = self.uninit(64 * 128)?;
        let rc = {
            // Keep the read guard alive until the asynchronous launch is queued.
            let sink_device = sink.map(|slice| slice.device_ptr(&stream));
            let sink_ptr = sink_device
                .as_ref()
                .map_or(std::ptr::null(), |(ptr, _)| *ptr as *const f32);
            unsafe {
                memra_mimo_sink_attn_decode_f32(
                    query.device_ptr(&stream).0 as *const f32,
                    key.device_ptr(&stream).0 as *const f32,
                    value.device_ptr(&stream).0 as *const f32,
                    sink_ptr,
                    output.device_ptr_mut(&stream).0 as *mut f32,
                    seq as i32,
                    64,
                    kv_heads as i32,
                    192,
                    128,
                    window as i32,
                    stream.cu_stream() as *mut c_void,
                )
            }
        };
        if rc != 0 {
            return Err(format!("MiMo attention GPU decode returned {rc}").into());
        }
        stream.synchronize()?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memra_gguf::config::{HfConfig, ModelConfig};
    use memra_gguf::model_plan::ModelPlan;

    const HEADS: usize = 64;
    const QK_DIM: usize = 192;
    const V_DIM: usize = 128;

    #[test]
    fn pinned_plan_rejects_wrong_sink_and_window() {
        let config = ModelConfig::from_hf(&HfConfig::parse(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../memra-gguf/src/model_packs/mimo_v2/fixtures/config.json"
        ))));
        let plan = ModelPlan::compile(&config).unwrap();
        let full = &plan.layers[0].attention;
        let AttentionPlan::Full(_) = full else {
            panic!("layer 0 must be full");
        };
        assert_eq!(validate_plan(full, false), Ok((4, 0)));
        assert!(validate_plan(full, true).is_err());
        let sliding = &plan.layers[1].attention;
        let AttentionPlan::SlidingWindow { .. } = sliding else {
            panic!("layer 1 must be sliding");
        };
        assert_eq!(validate_plan(sliding, true), Ok((8, 128)));
        assert!(validate_plan(sliding, false).is_err());
        let mut wrong_window = sliding.clone();
        if let AttentionPlan::SlidingWindow { window, .. } = &mut wrong_window {
            *window = 0;
        }
        assert!(validate_plan(&wrong_window, true).is_err());
    }

    // CPU contract oracle for future device comparisons. These tests check the
    // layout and semantics only; they never launch the CUDA implementation.
    fn reference(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        sink: Option<&[f32]>,
        seq: usize,
        kv_heads: usize,
        window: usize,
    ) -> Vec<f32> {
        assert!(seq > 0 && (kv_heads == 4 || kv_heads == 8));
        assert!(window == 0 || window == 128);
        assert_eq!(q.len(), HEADS * QK_DIM);
        assert_eq!(k.len(), seq * kv_heads * QK_DIM);
        assert_eq!(v.len(), seq * kv_heads * V_DIM);
        if let Some(sink) = sink {
            assert_eq!(sink.len(), HEADS);
        }

        let start = if window == 0 {
            0
        } else {
            seq.saturating_sub(window)
        };
        let scale = 1.0 / (QK_DIM as f32).sqrt();
        let mut output = vec![0.0; HEADS * V_DIM];
        for head in 0..HEADS {
            let kv_head = head / (HEADS / kv_heads);
            let mut logits = Vec::with_capacity(seq - start);
            for token in start..seq {
                let q_row = &q[head * QK_DIM..(head + 1) * QK_DIM];
                let k_base = (token * kv_heads + kv_head) * QK_DIM;
                let score: f32 = q_row
                    .iter()
                    .zip(&k[k_base..k_base + QK_DIM])
                    .map(|(a, b)| a * b)
                    .sum();
                logits.push(score * scale);
            }
            let max = logits
                .iter()
                .copied()
                .chain(sink.map(|s| s[head]))
                .fold(f32::NEG_INFINITY, f32::max);
            let sink_weight = sink.map_or(0.0, |s| (s[head] - max).exp());
            let denominator =
                sink_weight + logits.iter().map(|logit| (logit - max).exp()).sum::<f32>();
            for (offset, logit) in logits.iter().enumerate() {
                let v_base = ((start + offset) * kv_heads + kv_head) * V_DIM;
                let weight = (logit - max).exp() / denominator;
                for dim in 0..V_DIM {
                    output[head * V_DIM + dim] += weight * v[v_base + dim];
                }
            }
        }
        output
    }

    #[test]
    fn grouped_heads_and_optional_sink() {
        for kv_heads in [4, 8] {
            let q = vec![0.0; HEADS * QK_DIM];
            let k = vec![0.0; 2 * kv_heads * QK_DIM];
            let mut v = vec![0.0; 2 * kv_heads * V_DIM];
            for token in 0..2 {
                for kv_head in 0..kv_heads {
                    v[(token * kv_heads + kv_head) * V_DIM] =
                        10.0 * kv_head as f32 + 2.0 * token as f32;
                }
            }
            let plain = reference(&q, &k, &v, None, 2, kv_heads, 0);
            let sink = reference(&q, &k, &v, Some(&[0.0; HEADS]), 2, kv_heads, 0);
            for head in 0..HEADS {
                let kv_head = head / (HEADS / kv_heads);
                let expected_sum = 20.0 * kv_head as f32 + 2.0;
                assert!((plain[head * V_DIM] - expected_sum / 2.0).abs() < 1e-5);
                assert!((sink[head * V_DIM] - expected_sum / 3.0).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn window_excludes_old_keys() {
        let seq = 130;
        let kv_heads = 4;
        let q = vec![0.0; HEADS * QK_DIM];
        let k = vec![0.0; seq * kv_heads * QK_DIM];
        let mut v = vec![2.0; seq * kv_heads * V_DIM];
        for token in 0..2 {
            for kv_head in 0..kv_heads {
                v[(token * kv_heads + kv_head) * V_DIM] = 100.0;
            }
        }
        let full = reference(&q, &k, &v, None, seq, kv_heads, 0);
        let recent = reference(&q, &k, &v, None, seq, kv_heads, 128);
        assert!((full[0] - (200.0 + 128.0 * 2.0) / 130.0).abs() < 1e-4);
        assert!((recent[0] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn large_logits_keep_the_sink_in_the_denominator() {
        let kv_heads = 4;
        let mut q = vec![0.0; HEADS * QK_DIM];
        let mut k = vec![0.0; kv_heads * QK_DIM];
        let mut v = vec![0.0; kv_heads * V_DIM];
        q[0] = 100_000.0;
        k[0] = 1.0;
        v[0] = 2.0;
        let sink_logit = 100_000.0 / (QK_DIM as f32).sqrt();
        let mut sink = [0.0; HEADS];
        sink[0] = sink_logit;
        let output = reference(&q, &k, &v, Some(&sink), 1, kv_heads, 0);
        assert!(output[0].is_finite());
        assert!((output[0] - 1.0).abs() < 1e-5);
    }
}
