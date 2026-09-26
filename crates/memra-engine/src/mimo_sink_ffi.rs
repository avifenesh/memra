//! Standalone MiMo f32 sink-attention decode FFI.
//!
//! The CUDA translation unit is `cu/mimo_sink_attn.cu`. Integration must build
//! that unit and include this module before calling the declaration below.

use core::ffi::c_void;

unsafe extern "C" {
    /// Q [64,192], K [seq,kv_heads,192], V [seq,kv_heads,128],
    /// optional sink [64], output [64,128]. All non-null pointers address
    /// contiguous device f32 storage. `seq` includes the current token.
    ///
    /// `kv_heads` is 4 or 8. `window` is 0 for full history or 128 for the
    /// last min(seq,128) keys. Returns 0 on accepted asynchronous launch,
    /// 40001..40004 on invalid arguments, or 10000 + cudaError_t for a CUDA
    /// runtime or launch error. Synchronize the stream to catch execution
    /// failures. Callers must keep input and output storage alive until then.
    #[allow(dead_code)] // Kept standalone until the owner wires it into the engine.
    pub(crate) fn memra_mimo_sink_attn_decode_f32(
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

#[cfg(test)]
mod tests {
    const HEADS: usize = 64;
    const QK_DIM: usize = 192;
    const V_DIM: usize = 128;

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
