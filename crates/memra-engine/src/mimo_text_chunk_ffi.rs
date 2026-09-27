//! Bounded native MiMo text attention over one fresh causal chunk.
//! No preceding KV cache, persistent KV update, or serving dispatch is wired.

use core::ffi::c_void;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use memra_gguf::model_plan::AttentionPlan;
use memra_reference::mimo_text_chunk::{QK_DIM, QUERY_HEADS, VALUE_DIM, validate_request};

use crate::Engine;

unsafe extern "C" {
    fn memra_mimo_text_chunk_attention_f32(
        query: *const f32,
        key: *const f32,
        value: *const f32,
        sink: *const f32,
        output: *mut f32,
        chunk: i32,
        heads: i32,
        kv_heads: i32,
        qk_dim: i32,
        value_dim: i32,
        window: i32,
        stream: *mut c_void,
    ) -> i32;
}

impl Engine {
    /// Synchronous `[chunk,64,192]` Q, `[chunk,KV,192]` K,
    /// `[chunk,KV,128]` pre-scaled V to `[chunk,64,128]` f32 attention.
    ///
    /// The chunk begins at sequence position zero. Q/K must already be
    /// RoPE-rotated, and V must already include the source's 0.707 factor.
    /// Local layers use the learned sink only in the softmax denominator.
    /// Finite inputs and results are checked on the device before return.
    pub fn mimo_text_chunk_attention(
        &self,
        attention: &AttentionPlan,
        query: &CudaSlice<f32>,
        key: &CudaSlice<f32>,
        value: &CudaSlice<f32>,
        sink: Option<&CudaSlice<f32>>,
        chunk: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let shape = validate_request(
            attention,
            chunk,
            query.len(),
            key.len(),
            value.len(),
            sink.map(CudaSlice::len),
        )?;
        self.gpu.ctx.bind_to_thread()?;
        let stream = self.stream();
        let device = stream.context().ordinal();
        if query.ordinal() != device
            || key.ordinal() != device
            || value.ordinal() != device
            || sink.is_some_and(|s| s.ordinal() != device)
        {
            return Err("MiMo chunk Q/K/V or sink crossed GPU devices".into());
        }
        let mut output = self.uninit(chunk * QUERY_HEADS * VALUE_DIM)?;
        let rc = {
            let sink_device = sink.map(|s| s.device_ptr(&stream));
            let sink_ptr = sink_device
                .as_ref()
                .map_or(std::ptr::null(), |(ptr, _)| *ptr as *const f32);
            unsafe {
                memra_mimo_text_chunk_attention_f32(
                    query.device_ptr(&stream).0 as *const f32,
                    key.device_ptr(&stream).0 as *const f32,
                    value.device_ptr(&stream).0 as *const f32,
                    sink_ptr,
                    output.device_ptr_mut(&stream).0 as *mut f32,
                    chunk as i32,
                    QUERY_HEADS as i32,
                    shape.kv_heads as i32,
                    QK_DIM as i32,
                    VALUE_DIM as i32,
                    shape.window as i32,
                    stream.cu_stream() as *mut c_void,
                )
            }
        };
        if rc != 0 {
            return Err(format!("MiMo text chunk attention CUDA refusal {rc}").into());
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memra_gguf::config::{HfConfig, ModelConfig};
    use memra_gguf::model_plan::ModelPlan;
    use memra_reference::mimo_text_chunk::preprojected_attention;

    fn plans() -> (AttentionPlan, AttentionPlan) {
        let config = ModelConfig::from_hf(&HfConfig::parse(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../memra-gguf/src/model_packs/mimo_v2/fixtures/config.json"
        ))));
        let plan = ModelPlan::compile(&config).unwrap();
        (
            plan.layers[0].attention.clone(),
            plan.layers[1].attention.clone(),
        )
    }

    #[test]
    fn source_plan_rejects_wrong_chunk_shape() {
        let (global, local) = plans();
        assert!(
            validate_request(
                &global,
                256,
                256 * 64 * 192,
                256 * 4 * 192,
                256 * 4 * 128,
                None
            )
            .is_ok()
        );
        assert!(
            validate_request(
                &local,
                129,
                129 * 64 * 192,
                129 * 8 * 192,
                129 * 8 * 128,
                Some(64)
            )
            .is_ok()
        );
        assert!(validate_request(&global, 257, 0, 0, 0, None).is_err());
        assert!(validate_request(&local, 1, 64 * 192, 8 * 192, 8 * 128, Some(63)).is_err());
    }

    #[test]
    #[ignore = "requires a dedicated MiMo GPU component lane"]
    fn gpu_chunk_matches_portable_causal_oracle() -> Result<(), Box<dyn std::error::Error>> {
        let gpu: usize = std::env::var("MEMRA_MIMO_COMPONENT_GPU")
            .unwrap_or_else(|_| "0".into())
            .parse()?;
        let engine = Engine::new(gpu)?;
        let (global, local) = plans();
        for (plan, kv_heads) in [(global, 4), (local, 8)] {
            for chunk in [1, 9, 129, 256] {
                let q = (0..chunk * QUERY_HEADS * QK_DIM)
                    .map(|i| ((i * 11 % 71) as f32 - 35.0) / 53.0)
                    .collect::<Vec<_>>();
                let k = (0..chunk * kv_heads * QK_DIM)
                    .map(|i| ((i * 17 % 67) as f32 - 33.0) / 47.0)
                    .collect::<Vec<_>>();
                let v = (0..chunk * kv_heads * VALUE_DIM)
                    .map(|i| ((i * 19 % 61) as f32 - 30.0) / 57.0)
                    .collect::<Vec<_>>();
                let sink = (kv_heads == 8).then(|| {
                    (0..QUERY_HEADS)
                        .map(|i| (i as f32 - 31.0) / 13.0)
                        .collect::<Vec<_>>()
                });
                let expected = preprojected_attention(&plan, &q, &k, &v, sink.as_deref(), chunk)?;
                let q_gpu = engine.htod(&q)?;
                let k_gpu = engine.htod(&k)?;
                let v_gpu = engine.htod(&v)?;
                let sink_gpu = sink.as_ref().map(|s| engine.htod(s)).transpose()?;
                let actual = engine.mimo_text_chunk_attention(
                    &plan,
                    &q_gpu,
                    &k_gpu,
                    &v_gpu,
                    sink_gpu.as_ref(),
                    chunk,
                )?;
                let actual = engine.dtoh(&actual)?;
                assert_eq!(actual.len(), expected.len());
                for (i, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
                    assert!(
                        (got - want).abs() < 3e-4,
                        "mode KV{kv_heads}, chunk {chunk}, element {i}: {got} vs {want}"
                    );
                }
                let mut bad = q.clone();
                bad[0] = f32::NAN;
                let bad_gpu = engine.htod(&bad)?;
                assert!(
                    engine
                        .mimo_text_chunk_attention(
                            &plan,
                            &bad_gpu,
                            &k_gpu,
                            &v_gpu,
                            sink_gpu.as_ref(),
                            chunk,
                        )
                        .is_err()
                );
            }
        }
        Ok(())
    }
}
