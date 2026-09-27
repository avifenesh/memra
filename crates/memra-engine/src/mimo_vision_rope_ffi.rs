//! Bounded MiMo vision two-axis RoPE after source QKV projection. The caller
//! owns patch projection, QKV split/bias, and visual attention.

use core::ffi::c_void;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use memra_gguf::model_packs::mimo_v2::vision::{MiMoVisionAttentionPlan, MiMoVisionLayout};
use memra_reference::mimo_vision_rope::phase_rows;

use crate::Engine;

type Fail = Box<dyn std::error::Error>;
const QUERY_WIDTH: usize = 32 * 64;
const KEY_WIDTH: usize = 8 * 64;
const HALF: usize = 32;

unsafe extern "C" {
    fn memra_mimo_vision_rope_f32(
        q: *const f32,
        k: *const f32,
        cos: *const f32,
        sin: *const f32,
        out_q: *mut f32,
        out_k: *mut f32,
        patches: i32,
        query_heads: i32,
        key_heads: i32,
        head_dim: i32,
        stream: *mut c_void,
    ) -> i32;
}

pub struct MiMoVisionRotatedQk {
    pub query: CudaSlice<f32>,
    pub key: CudaSlice<f32>,
}

impl Engine {
    /// Apply publisher axial height/width RoPE in this block's patch order.
    /// Q/K are f32 views of BF16 source projections and return BF16-rounded
    /// f32 values, matching the source's `to(orig_dtype)` boundary.
    pub fn mimo_vision_axial_rope(
        &self,
        plan: &MiMoVisionAttentionPlan,
        layout: &MiMoVisionLayout,
        query: &CudaSlice<f32>,
        key: &CudaSlice<f32>,
    ) -> Result<MiMoVisionRotatedQk, Fail> {
        let (cos, sin) = phase_rows(plan, layout)?;
        let patches = cos.len() / HALF;
        if cos.len() != patches * HALF
            || sin.len() != cos.len()
            || query.len() != patches * QUERY_WIDTH
            || key.len() != patches * KEY_WIDTH
            || cos.iter().chain(&sin).any(|value| !value.is_finite())
        {
            return Err("MiMo vision axial RoPE phase or Q/K extent changed".into());
        }
        self.gpu.ctx.bind_to_thread()?;
        let stream = self.stream();
        let device = stream.context().ordinal();
        if query.ordinal() != device || key.ordinal() != device {
            return Err("MiMo vision axial Q/K crossed GPU devices".into());
        }
        let cos = self.htod(&cos)?;
        let sin = self.htod(&sin)?;
        let mut out_q = self.uninit(query.len())?;
        let mut out_k = self.uninit(key.len())?;
        let rc = {
            let (q_ptr, q_guard) = query.device_ptr(&stream);
            let (k_ptr, k_guard) = key.device_ptr(&stream);
            let (c_ptr, c_guard) = cos.device_ptr(&stream);
            let (s_ptr, s_guard) = sin.device_ptr(&stream);
            let (oq_ptr, oq_guard) = out_q.device_ptr_mut(&stream);
            let (ok_ptr, ok_guard) = out_k.device_ptr_mut(&stream);
            let code = unsafe {
                memra_mimo_vision_rope_f32(
                    q_ptr as *const f32,
                    k_ptr as *const f32,
                    c_ptr as *const f32,
                    s_ptr as *const f32,
                    oq_ptr as *mut f32,
                    ok_ptr as *mut f32,
                    patches as i32,
                    32,
                    8,
                    64,
                    stream.cu_stream() as *mut c_void,
                )
            };
            drop((q_guard, k_guard, c_guard, s_guard, oq_guard, ok_guard));
            code
        };
        if rc != 0 {
            return Err(format!("MiMo vision axial RoPE CUDA refusal {rc}").into());
        }
        stream.synchronize()?;
        Ok(MiMoVisionRotatedQk {
            query: out_q,
            key: out_k,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memra_gguf::config::{HfConfig, ModelConfig};
    use memra_gguf::model_packs::mimo_v2::vision::{
        MiMoVisionGrid, pinned_attention_plan, pinned_vision_layout,
    };
    use memra_reference::mimo_vision_rope::rotate_qk;

    #[test]
    #[ignore = "requires a dedicated MiMo GPU component lane"]
    fn gpu_axial_rope_matches_bf16_portable_reference() -> Result<(), Fail> {
        let gpu: usize = std::env::var("MEMRA_MIMO_COMPONENT_GPU")
            .unwrap_or_else(|_| "0".into())
            .parse()?;
        let engine = Engine::new(gpu)?;
        let config = ModelConfig::from_hf(&HfConfig::parse(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../memra-gguf/src/model_packs/mimo_v2/fixtures/config.json"
        ))));
        let layout = pinned_vision_layout(
            &config,
            &[
                MiMoVisionGrid {
                    frames: 1,
                    height: 2,
                    width: 2,
                },
                MiMoVisionGrid {
                    frames: 1,
                    height: 4,
                    width: 6,
                },
            ],
        )?;
        let patches = layout.row_positions.len();
        let q = (0..patches * QUERY_WIDTH)
            .map(|index| ((index * 7 % 61) as f32 - 30.0) / 17.0)
            .collect::<Vec<_>>();
        let k = (0..patches * KEY_WIDTH)
            .map(|index| ((index * 11 % 67) as f32 - 33.0) / 19.0)
            .collect::<Vec<_>>();
        for layer in [0, 5] {
            let plan = pinned_attention_plan(&config, layer)?;
            let (expected_q, expected_k) = rotate_qk(&plan, &layout, &q, &k)?;
            let actual = engine.mimo_vision_axial_rope(
                &plan,
                &layout,
                &engine.htod(&q)?,
                &engine.htod(&k)?,
            )?;
            let actual_q = engine.dtoh(&actual.query)?;
            let actual_k = engine.dtoh(&actual.key)?;
            for (name, got, want) in [
                ("query", actual_q.as_slice(), expected_q.as_slice()),
                ("key", actual_k.as_slice(), expected_k.as_slice()),
            ] {
                let maximum = got
                    .iter()
                    .zip(want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                eprintln!("MiMo vision layer {layer} {name} axial RoPE maxabs {maximum}");
                assert!(maximum <= 0.01, "{name} axial RoPE maxabs {maximum}");
            }
        }
        Ok(())
    }
}
