//! Bounded MiMo audio patch attention after QKV projection and Q/K RoPE.
//! This is a six-layer encoder component, not an audio or serving path.

use core::ffi::c_void;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use memra_gguf::model_packs::mimo_v2::audio::MiMoAudioPatchPlan;
use memra_reference::mimo_audio_attn::validate_request;

use crate::Engine;

unsafe extern "C" {
    fn memra_mimo_audio_preprojected_f32(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        output: *mut f32,
        layer: i32,
        groups: i32,
        group_size: i32,
        heads: i32,
        head_dim: i32,
        full_attention: i32,
        stream: *mut c_void,
    ) -> i32;
}

impl Engine {
    /// Synchronous f32 `[groups,4,16,64]` Q/K/V to the same-shaped result.
    /// Q/K must already be RoPE-rotated. Each four-token group is independent
    /// and fully visible; no KV state is read or written. Finite inputs and
    /// outputs are checked on device before this method returns.
    pub fn mimo_audio_preprojected_attention(
        &self,
        plan: &MiMoAudioPatchPlan,
        layer: usize,
        query: &CudaSlice<f32>,
        key: &CudaSlice<f32>,
        value: &CudaSlice<f32>,
        groups: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let elements = validate_request(plan, layer, groups, query.len(), key.len(), value.len())?;
        self.gpu.ctx.bind_to_thread()?;
        let stream = self.stream();
        let device = stream.context().ordinal();
        if query.ordinal() != device || key.ordinal() != device || value.ordinal() != device {
            return Err("MiMo audio Q/K/V crossed GPU devices".into());
        }
        let mut output = self.uninit(elements)?;
        let (q_ptr, q_guard) = query.device_ptr(&stream);
        let (k_ptr, k_guard) = key.device_ptr(&stream);
        let (v_ptr, v_guard) = value.device_ptr(&stream);
        let (out_ptr, out_guard) = output.device_ptr_mut(&stream);
        let rc = unsafe {
            memra_mimo_audio_preprojected_f32(
                q_ptr as *const f32,
                k_ptr as *const f32,
                v_ptr as *const f32,
                out_ptr as *mut f32,
                layer as i32,
                groups as i32,
                plan.group_size as i32,
                plan.local_heads as i32,
                plan.local_head_dim as i32,
                i32::from(plan.local_full_attention),
                stream.cu_stream() as *mut c_void,
            )
        };
        drop((q_guard, k_guard, v_guard, out_guard));
        if rc != 0 {
            return Err(format!("MiMo audio attention CUDA refusal {rc}").into());
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memra_gguf::config::{HfConfig, ModelConfig};
    use memra_gguf::model_packs::mimo_v2::audio::pinned_patch_plan;
    use memra_reference::mimo_audio_attn::preprojected_attention;

    fn plan() -> MiMoAudioPatchPlan {
        let config = ModelConfig::from_hf(&HfConfig::parse(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../memra-gguf/src/model_packs/mimo_v2/fixtures/config.json"
        ))));
        pinned_patch_plan(&config).unwrap()
    }

    #[test]
    fn gpu_boundary_requires_exact_six_layer_four_token_geometry() {
        let program = plan();
        assert_eq!(
            validate_request(&program, 5, 1, 4 * 16 * 64, 4 * 16 * 64, 4 * 16 * 64),
            Ok(4 * 16 * 64)
        );
        assert!(validate_request(&program, 6, 1, 4 * 16 * 64, 4 * 16 * 64, 4 * 16 * 64).is_err());
        assert!(validate_request(&program, 0, 1_501, 0, 0, 0).is_err());
        assert!(
            validate_request(&program, 0, 1, 4 * 16 * 64 - 1, 4 * 16 * 64, 4 * 16 * 64).is_err()
        );
        let mut causal = program;
        causal.local_full_attention = false;
        assert!(validate_request(&causal, 0, 1, 4 * 16 * 64, 4 * 16 * 64, 4 * 16 * 64).is_err());
    }

    #[test]
    #[ignore = "requires a dedicated MiMo GPU component lane"]
    fn gpu_preprojected_attention_matches_f32_reference() -> Result<(), Box<dyn std::error::Error>>
    {
        let gpu: usize = std::env::var("MEMRA_MIMO_COMPONENT_GPU")
            .unwrap_or_else(|_| "0".into())
            .parse()?;
        let engine = Engine::new(gpu)?;
        let program = plan();
        let groups = 3usize;
        let elements = groups * 4 * 16 * 64;
        let q = (0..elements)
            .map(|index| ((index * 7 % 29) as f32 - 14.0) / 13.0)
            .collect::<Vec<_>>();
        let k = (0..elements)
            .map(|index| ((index * 11 % 31) as f32 - 15.0) / 17.0)
            .collect::<Vec<_>>();
        let v = (0..elements)
            .map(|index| ((index * 13 % 37) as f32 - 18.0) / 19.0)
            .collect::<Vec<_>>();
        let expected = preprojected_attention(&program, 2, &q, &k, &v, groups)?;
        let actual = engine.mimo_audio_preprojected_attention(
            &program,
            2,
            &engine.htod(&q)?,
            &engine.htod(&k)?,
            &engine.htod(&v)?,
            groups,
        )?;
        let actual = engine.dtoh(&actual)?;
        assert_eq!(actual.len(), expected.len());
        for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (got - want).abs() < 2e-5,
                "element {index}: {got} != {want}"
            );
        }
        Ok(())
    }
}
