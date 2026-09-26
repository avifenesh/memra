//! Bounded GPU attention over MiMo ViT Q/K/V after projection and axial RoPE.
//! Patch embedding, projection, RoPE, the ViT block, merger, and model serving
//! have separate admission gates.

use core::ffi::c_void;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use memra_gguf::model_packs::mimo_v2::vision::{MiMoPatchOrder, MiMoVisionAttentionPlan};

use crate::Engine;

pub const MIMO_VISION_MAX_SEQUENCE_PATCHES: usize = 256;
pub const MIMO_VISION_MAX_TOTAL_PATCHES: usize = 1024;
pub const MIMO_VISION_MAX_SEQUENCES: usize = 32;
const QUERY_HEADS: usize = 32;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 64;
const QUERY_WIDTH: usize = QUERY_HEADS * HEAD_DIM;
const KV_WIDTH: usize = KV_HEADS * HEAD_DIM;

unsafe extern "C" {
    fn memra_mimo_vision_preprojected_f32(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        sink: *const f32,
        output: *mut f32,
        lengths: *const i32,
        sequence_count: i32,
        patches: i32,
        query_heads: i32,
        kv_heads: i32,
        head_dim: i32,
        window: i32,
        stream: *mut c_void,
    ) -> i32;
}

#[derive(Debug, PartialEq, Eq)]
struct ValidatedRequest {
    lengths: Vec<i32>,
    patches: usize,
    window: i32,
}

fn validate_plan(plan: &MiMoVisionAttentionPlan, has_sink: bool) -> Result<i32, &'static str> {
    if plan.layer >= 28
        || plan.query_heads != QUERY_HEADS
        || plan.kv_heads != KV_HEADS
        || plan.head_dim != HEAD_DIM
    {
        return Err("MiMo vision attention differs from pinned Q32/KV8/D64 geometry");
    }
    let full = [0, 9, 18, 27].contains(&plan.layer);
    let expected_column = [5, 6, 7, 8, 14, 15, 16, 17, 23, 24, 25, 26].contains(&plan.layer);
    if plan.symmetric_window != (!full).then_some(64)
        || plan.sink_first_key != !full
        || has_sink != !full
        || (plan.patch_order == MiMoPatchOrder::Column) != expected_column
    {
        return Err("MiMo vision attention layer, window, first-key bias, or patch order changed");
    }
    Ok(if full { 0 } else { 64 })
}

fn validate_request(
    plan: &MiMoVisionAttentionPlan,
    lengths: &[usize],
    query_len: usize,
    key_len: usize,
    value_len: usize,
    sink_len: Option<usize>,
) -> Result<ValidatedRequest, &'static str> {
    let window = validate_plan(plan, sink_len.is_some())?;
    if lengths.is_empty() || lengths.len() > MIMO_VISION_MAX_SEQUENCES {
        return Err("MiMo vision sequence count exceeds MIMO_VISION_MAX_SEQUENCES");
    }
    let mut patches = 0usize;
    let mut packed = Vec::with_capacity(lengths.len());
    for &length in lengths {
        if length == 0 || length > MIMO_VISION_MAX_SEQUENCE_PATCHES {
            return Err("MiMo vision sequence exceeds MIMO_VISION_MAX_SEQUENCE_PATCHES");
        }
        patches = patches
            .checked_add(length)
            .ok_or("MiMo vision patch count overflow")?;
        packed.push(length as i32);
    }
    if patches > MIMO_VISION_MAX_TOTAL_PATCHES {
        return Err("MiMo vision patches exceed MIMO_VISION_MAX_TOTAL_PATCHES");
    }
    let expected_query = patches
        .checked_mul(QUERY_WIDTH)
        .ok_or("MiMo vision query extent overflow")?;
    let expected_kv = patches
        .checked_mul(KV_WIDTH)
        .ok_or("MiMo vision KV extent overflow")?;
    if query_len != expected_query
        || key_len != expected_kv
        || value_len != expected_kv
        || sink_len.is_some_and(|len| len != QUERY_HEADS)
    {
        return Err("MiMo vision Q/K/V or first-key bias extent mismatch");
    }
    Ok(ValidatedRequest {
        lengths: packed,
        patches,
        window,
    })
}

impl Engine {
    /// Synchronous, bounded component over contiguous f32 device tensors.
    /// Q/K must already be axial-RoPE rotated and all tensors must follow the
    /// plan's patch order. `lengths` splits independent images or frames.
    /// Finite inputs and finite outputs are checked on the device before this
    /// method returns. It does not perform the attention output projection.
    pub fn mimo_vision_preprojected_attention(
        &self,
        plan: &MiMoVisionAttentionPlan,
        query: &CudaSlice<f32>,
        key: &CudaSlice<f32>,
        value: &CudaSlice<f32>,
        lengths: &[usize],
        first_key_bias: Option<&CudaSlice<f32>>,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let request = validate_request(
            plan,
            lengths,
            query.len(),
            key.len(),
            value.len(),
            first_key_bias.map(|bias| bias.len()),
        )?;
        self.gpu.ctx.bind_to_thread()?;
        let stream = self.stream();
        let device = stream.context().ordinal();
        if query.ordinal() != device
            || key.ordinal() != device
            || value.ordinal() != device
            || first_key_bias.is_some_and(|bias| bias.ordinal() != device)
        {
            return Err("MiMo vision Q/K/V or first-key bias crossed GPU devices".into());
        }
        let mut output = self.uninit(request.patches * QUERY_WIDTH)?;
        let rc = {
            let bias_device = first_key_bias.map(|bias| bias.device_ptr(&stream));
            let bias_ptr = bias_device
                .as_ref()
                .map_or(std::ptr::null(), |(ptr, _)| *ptr as *const f32);
            unsafe {
                memra_mimo_vision_preprojected_f32(
                    query.device_ptr(&stream).0 as *const f32,
                    key.device_ptr(&stream).0 as *const f32,
                    value.device_ptr(&stream).0 as *const f32,
                    bias_ptr,
                    output.device_ptr_mut(&stream).0 as *mut f32,
                    request.lengths.as_ptr(),
                    request.lengths.len() as i32,
                    request.patches as i32,
                    QUERY_HEADS as i32,
                    KV_HEADS as i32,
                    HEAD_DIM as i32,
                    request.window,
                    stream.cu_stream() as *mut c_void,
                )
            }
        };
        if rc != 0 {
            return Err(format!("MiMo vision attention CUDA refusal {rc}").into());
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memra_gguf::config::{HfConfig, ModelConfig};
    use memra_gguf::model_packs::mimo_v2::vision::pinned_attention_plan;
    use memra_reference::mimo_vision::preprojected_attention;

    fn plan(layer: u32) -> MiMoVisionAttentionPlan {
        let config = ModelConfig::from_hf(&HfConfig::parse(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../memra-gguf/src/model_packs/mimo_v2/fixtures/config.json"
        ))));
        pinned_attention_plan(&config, layer).unwrap()
    }

    #[test]
    fn pinned_layers_preserve_geometry_window_bias_and_patch_order() {
        for layer in 0..28 {
            let program = plan(layer);
            let full = [0, 9, 18, 27].contains(&layer);
            assert_eq!(
                validate_plan(&program, !full),
                Ok(if full { 0 } else { 64 })
            );
            assert!(validate_plan(&program, full).is_err());
        }
        let mut changed = plan(1);
        changed.query_heads = 16;
        assert!(validate_plan(&changed, true).is_err());
        changed = plan(1);
        changed.symmetric_window = Some(32);
        assert!(validate_plan(&changed, true).is_err());
        changed = plan(5);
        changed.patch_order = MiMoPatchOrder::Row;
        assert!(validate_plan(&changed, true).is_err());
        changed = plan(1);
        changed.layer = 28;
        assert!(validate_plan(&changed, true).is_err());
    }

    #[test]
    fn lengths_and_extents_are_bounded_before_gpu_work() {
        let local = plan(1);
        let valid = validate_request(
            &local,
            &[2, 3],
            5 * QUERY_WIDTH,
            5 * KV_WIDTH,
            5 * KV_WIDTH,
            Some(32),
        )
        .unwrap();
        assert_eq!(valid.lengths, vec![2, 3]);
        assert_eq!(valid.patches, 5);
        assert_eq!(valid.window, 64);
        assert!(validate_request(&local, &[], 0, 0, 0, Some(32)).is_err());
        assert!(validate_request(&local, &[0], 0, 0, 0, Some(32)).is_err());
        assert!(validate_request(&local, &[257], 0, 0, 0, Some(32)).is_err());
        assert!(validate_request(&local, &[1; 33], 0, 0, 0, Some(32)).is_err());
        assert!(validate_request(&local, &[256; 5], 0, 0, 0, Some(32)).is_err());
        assert!(
            validate_request(
                &local,
                &[5],
                4 * QUERY_WIDTH,
                5 * KV_WIDTH,
                5 * KV_WIDTH,
                Some(32)
            )
            .is_err()
        );
        assert!(
            validate_request(
                &local,
                &[5],
                5 * QUERY_WIDTH,
                5 * KV_WIDTH,
                5 * KV_WIDTH,
                Some(31)
            )
            .is_err()
        );
        assert!(
            validate_request(
                &local,
                &[5],
                5 * QUERY_WIDTH,
                5 * KV_WIDTH,
                5 * KV_WIDTH,
                None
            )
            .is_err()
        );
        let full = plan(0);
        assert!(validate_request(&full, &[1], QUERY_WIDTH, KV_WIDTH, KV_WIDTH, None).is_ok());
        assert!(validate_request(&full, &[1], QUERY_WIDTH, KV_WIDTH, KV_WIDTH, Some(32)).is_err());
    }

    #[test]
    #[ignore = "requires a dedicated MiMo GPU component lane"]
    fn gpu_visual_attention_matches_source_math_reference() -> Result<(), Box<dyn std::error::Error>>
    {
        let gpu: usize = std::env::var("MEMRA_MIMO_COMPONENT_GPU")
            .unwrap_or_else(|_| "0".into())
            .parse()?;
        let engine = Engine::new(gpu)?;
        for (layer, lengths) in [(0, vec![16, 7]), (1, vec![66]), (5, vec![33, 12])] {
            let plan = plan(layer);
            let patches: usize = lengths.iter().sum();
            let query = (0..patches * QUERY_WIDTH)
                .map(|index| ((index * 7 + 3) % 101) as f32 * 0.0125 - 0.625)
                .collect::<Vec<_>>();
            let key = (0..patches * KV_WIDTH)
                .map(|index| ((index * 11 + 5) % 97) as f32 * 0.01 - 0.48)
                .collect::<Vec<_>>();
            let value = (0..patches * KV_WIDTH)
                .map(|index| ((index * 13 + 7) % 89) as f32 * 0.015 - 0.66)
                .collect::<Vec<_>>();
            let sink = plan.sink_first_key.then(|| {
                (0..QUERY_HEADS)
                    .map(|head| head as f32 * 0.02 - 0.31)
                    .collect::<Vec<_>>()
            });
            let reference =
                preprojected_attention(&plan, &query, &key, &value, &lengths, sink.as_deref())?;
            let query_dev = engine.htod(&query)?;
            let key_dev = engine.htod(&key)?;
            let value_dev = engine.htod(&value)?;
            let sink_dev = sink.as_ref().map(|bias| engine.htod(bias)).transpose()?;
            let output = engine.mimo_vision_preprojected_attention(
                &plan,
                &query_dev,
                &key_dev,
                &value_dev,
                &lengths,
                sink_dev.as_ref(),
            )?;
            let actual = engine.dtoh(&output)?;
            assert_eq!(actual.len(), reference.len());
            let max_abs = actual
                .iter()
                .zip(&reference)
                .map(|(got, want)| (got - want).abs())
                .fold(0.0f32, f32::max);
            let rms = (actual
                .iter()
                .zip(&reference)
                .map(|(got, want)| (got - want).powi(2))
                .sum::<f32>()
                / actual.len() as f32)
                .sqrt();
            println!(
                "mimo_vision_gpu_parity gpu={gpu} layer={layer} patches={patches} max_abs={max_abs:.9} rms={rms:.9}"
            );
            assert!(
                max_abs <= 1e-4,
                "MiMo visual attention exceeded f32 parity bound"
            );
        }
        Ok(())
    }
}
