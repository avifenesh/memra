//! Grouped-code forward for the pinned MiMo audio patch encoder.
//!
//! This starts after the separate RVQ codec. It produces BF16-valued f32
//! `[groups, 4096]` rows for later text embedding replacement.

use core::ffi::c_void;
use std::error::Error;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use memra_gguf::model_packs::mimo_v2::audio::MiMoAudioPatchPlan;

use crate::Engine;
use crate::mimo_audio_patch_load::{MiMoAudioLocalLayer, MiMoAudioPatchWeights};
use crate::model::GpuTensor;

type Fail = Box<dyn Error>;

const CHANNELS: usize = 20;
const GROUP: usize = 4;
const HIDDEN: usize = 1_024;
const HEADS: usize = 16;
const HEAD_DIM: usize = 64;
const LAYERS: usize = 6;
const FF: usize = 4_096;
const PROJECTION_FF: usize = 16_384;
const OUTPUT: usize = 4_096;
const MAX_GROUPS: usize = 1_500;
// Qwen2Config.rms_norm_eps when the publisher creates its local Qwen2Config
// without an explicit override.
const RMS_EPSILON: f32 = 1e-6;

#[derive(Clone, Copy)]
#[repr(i32)]
enum Epilogue {
    Round = 0,
    Bias = 1,
    Residual = 2,
    SiluMul = 3,
    Gelu = 4,
    NormMul = 5,
}

unsafe extern "C" {
    fn memra_mimo_audio_patch_epilogue(
        a: *const f32,
        b: *const f32,
        output: *mut f32,
        elements: i32,
        width: i32,
        operation: i32,
        stream: *mut c_void,
    ) -> i32;
    fn memra_mimo_audio_patch_rope_bf16(
        query: *mut f32,
        key: *mut f32,
        groups: i32,
        stream: *mut c_void,
    ) -> i32;
    fn memra_mimo_audio_patch_check_finite(
        input: *const f32,
        elements: i32,
        fault: *mut i32,
        stream: *mut c_void,
    ) -> i32;
}

fn pinned_forward_plan(plan: &MiMoAudioPatchPlan, groups: usize) -> Result<usize, Fail> {
    if plan.code_channels != CHANNELS
        || plan.group_size != GROUP
        || plan.code_vocab != 1_280
        || plan.local_hidden != HIDDEN
        || plan.local_layers != LAYERS
        || plan.local_heads != HEADS
        || plan.local_head_dim != HEAD_DIM
        || plan.local_intermediate != FF
        || !plan.local_full_attention
        || plan.local_rope_theta.to_bits() != 640_000.0f32.to_bits()
        || plan.projection_input != GROUP * HIDDEN
        || plan.projection_intermediate != PROJECTION_FF
        || plan.output_hidden != OUTPUT
    {
        return Err("MiMo audio patch forward plan differs from pinned source".into());
    }
    if !(1..=MAX_GROUPS).contains(&groups) {
        return Err("MiMo audio patch forward groups are outside 1..=1500".into());
    }
    Ok(groups * GROUP)
}

fn check_matrix(
    matrix: &GpuTensor,
    input: usize,
    output: usize,
    ordinal: usize,
) -> Result<(), Fail> {
    if !matches!(matrix, GpuTensor::FloatBf16 { .. })
        || matrix.in_features() != input
        || matrix.out_features() != output
        || matrix.ordinal() != ordinal
    {
        return Err("MiMo audio patch matrix dtype, shape, or GPU changed".into());
    }
    Ok(())
}

fn check_vector(vector: &CudaSlice<f32>, width: usize, ordinal: usize) -> Result<(), Fail> {
    if vector.len() != width || vector.ordinal() != ordinal {
        return Err("MiMo audio patch vector shape or GPU changed".into());
    }
    Ok(())
}

fn validate_weights(weights: &MiMoAudioPatchWeights, ordinal: usize) -> Result<(), Fail> {
    if weights.local_layers.len() != LAYERS
        || weights.speech_embeddings.len() != CHANNELS
        || weights.speech_ptrs.len() != CHANNELS
        || weights.speech_ptrs.ordinal() != ordinal
    {
        return Err("MiMo audio patch resident tensor count or GPU changed".into());
    }
    for table in &weights.speech_embeddings {
        check_matrix(table, HIDDEN, 1_280, ordinal)?;
    }
    for layer in &weights.local_layers {
        validate_layer(layer, ordinal)?;
    }
    check_vector(&weights.final_norm, HIDDEN, ordinal)?;
    check_matrix(
        &weights.projection_in,
        GROUP * HIDDEN,
        PROJECTION_FF,
        ordinal,
    )?;
    check_matrix(&weights.projection_out, PROJECTION_FF, OUTPUT, ordinal)?;
    Ok(())
}

fn validate_layer(layer: &MiMoAudioLocalLayer, ordinal: usize) -> Result<(), Fail> {
    for vector in [
        &layer.input_norm,
        &layer.post_attention_norm,
        &layer.query_bias,
        &layer.key_bias,
        &layer.value_bias,
    ] {
        check_vector(vector, HIDDEN, ordinal)?;
    }
    for matrix in [
        &layer.query,
        &layer.key,
        &layer.value,
        &layer.attention_output,
    ] {
        check_matrix(matrix, HIDDEN, HIDDEN, ordinal)?;
    }
    check_matrix(&layer.mlp_gate, HIDDEN, FF, ordinal)?;
    check_matrix(&layer.mlp_up, HIDDEN, FF, ordinal)?;
    check_matrix(&layer.mlp_down, FF, HIDDEN, ordinal)?;
    Ok(())
}

fn epilogue(
    engine: &Engine,
    a: &CudaSlice<f32>,
    b: Option<&CudaSlice<f32>>,
    width: usize,
    operation: Epilogue,
) -> Result<CudaSlice<f32>, Fail> {
    engine.gpu.ctx.bind_to_thread()?;
    let stream = engine.stream();
    let ordinal = stream.context().ordinal();
    if a.is_empty()
        || width == 0
        || !a.len().is_multiple_of(width)
        || a.ordinal() != ordinal
        || a.len() > i32::MAX as usize
        || width > i32::MAX as usize
    {
        return Err("MiMo audio patch epilogue input extent or GPU changed".into());
    }
    let want_b = match operation {
        Epilogue::Round | Epilogue::Gelu => None,
        Epilogue::Bias | Epilogue::NormMul => Some(width),
        Epilogue::Residual | Epilogue::SiluMul => Some(a.len()),
    };
    if b.map(CudaSlice::len) != want_b || b.is_some_and(|rhs| rhs.ordinal() != ordinal) {
        return Err("MiMo audio patch epilogue second operand changed".into());
    }
    let mut output = engine.uninit(a.len())?;
    let (a_ptr, a_guard) = a.device_ptr(&stream);
    let (b_ptr, b_guard) = if let Some(rhs) = b {
        let (ptr, guard) = rhs.device_ptr(&stream);
        (ptr as *const f32, Some(guard))
    } else {
        (std::ptr::null(), None)
    };
    let (output_ptr, output_guard) = output.device_ptr_mut(&stream);
    let rc = unsafe {
        memra_mimo_audio_patch_epilogue(
            a_ptr as *const f32,
            b_ptr,
            output_ptr as *mut f32,
            a.len() as i32,
            width as i32,
            operation as i32,
            stream.cu_stream() as *mut c_void,
        )
    };
    drop((a_guard, b_guard, output_guard));
    if rc != 0 {
        return Err(format!("MiMo audio patch BF16 epilogue returned {rc}").into());
    }
    Ok(output)
}

fn round(engine: &Engine, x: &CudaSlice<f32>, width: usize) -> Result<CudaSlice<f32>, Fail> {
    epilogue(engine, x, None, width, Epilogue::Round)
}

fn linear(
    engine: &Engine,
    weight: &GpuTensor,
    input: &CudaSlice<f32>,
    rows: usize,
) -> Result<CudaSlice<f32>, Fail> {
    let projected = engine.matmul(weight, input, rows)?;
    round(engine, &projected, weight.out_features())
}

fn linear_bias(
    engine: &Engine,
    weight: &GpuTensor,
    bias: &CudaSlice<f32>,
    input: &CudaSlice<f32>,
    rows: usize,
) -> Result<CudaSlice<f32>, Fail> {
    let projected = engine.matmul(weight, input, rows)?;
    epilogue(
        engine,
        &projected,
        Some(bias),
        weight.out_features(),
        Epilogue::Bias,
    )
}

fn norm(
    engine: &Engine,
    input: &CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    unit_weight: &CudaSlice<f32>,
    rows: usize,
) -> Result<CudaSlice<f32>, Fail> {
    let mut result = engine.uninit(rows * HIDDEN)?;
    engine.rms_norm(input, unit_weight, &mut result, HIDDEN, rows, RMS_EPSILON)?;
    epilogue(engine, &result, Some(weight), HIDDEN, Epilogue::NormMul)
}

fn rope(
    engine: &Engine,
    query: &mut CudaSlice<f32>,
    key: &mut CudaSlice<f32>,
    groups: usize,
) -> Result<(), Fail> {
    engine.gpu.ctx.bind_to_thread()?;
    let stream = engine.stream();
    let ordinal = stream.context().ordinal();
    let elements = groups * GROUP * HIDDEN;
    if query.len() != elements
        || key.len() != elements
        || query.ordinal() != ordinal
        || key.ordinal() != ordinal
    {
        return Err("MiMo audio patch rotary Q/K extent or GPU changed".into());
    }
    let (q_ptr, q_guard) = query.device_ptr_mut(&stream);
    let (k_ptr, k_guard) = key.device_ptr_mut(&stream);
    let rc = unsafe {
        memra_mimo_audio_patch_rope_bf16(
            q_ptr as *mut f32,
            k_ptr as *mut f32,
            groups as i32,
            stream.cu_stream() as *mut c_void,
        )
    };
    drop((q_guard, k_guard));
    if rc != 0 {
        return Err(format!("MiMo audio patch BF16 RoPE returned {rc}").into());
    }
    Ok(())
}

fn ensure_finite(engine: &Engine, values: &CudaSlice<f32>) -> Result<(), Fail> {
    engine.gpu.ctx.bind_to_thread()?;
    let stream = engine.stream();
    if values.is_empty()
        || values.len() > i32::MAX as usize
        || values.ordinal() != stream.context().ordinal()
    {
        return Err("MiMo audio patch final output extent or GPU changed".into());
    }
    let mut fault = engine.htod_i32(&[0])?;
    let (input_ptr, input_guard) = values.device_ptr(&stream);
    let (fault_ptr, fault_guard) = fault.device_ptr_mut(&stream);
    let rc = unsafe {
        memra_mimo_audio_patch_check_finite(
            input_ptr as *const f32,
            values.len() as i32,
            fault_ptr as *mut i32,
            stream.cu_stream() as *mut c_void,
        )
    };
    drop((input_guard, fault_guard));
    if rc != 0 || engine.dtoh_i32(&fault)? != [0] {
        return Err(format!("MiMo audio patch final output is not finite (CUDA rc {rc})").into());
    }
    Ok(())
}

impl MiMoAudioPatchWeights {
    /// Execute the pinned audio patch encoder on already grouped
    /// `[groups, 4, 20]` code IDs. Each group becomes one BF16-valued
    /// `[4096]` trunk row. No codec, placeholder insertion, or serving path.
    pub fn forward_grouped_codes(
        &self,
        engine: &Engine,
        codes: &[u16],
        groups: usize,
    ) -> Result<CudaSlice<f32>, Fail> {
        let rows = pinned_forward_plan(&self.plan, groups)?;
        validate_weights(self, engine.stream().context().ordinal())?;
        if codes.len() != rows * CHANNELS || codes.iter().any(|&code| code >= 1_280) {
            return Err("MiMo audio patch grouped code extent or ID changed".into());
        }
        let mut hidden = self.sum_grouped_codes(engine, codes, groups)?;
        if hidden.len() != rows * HIDDEN {
            return Err("MiMo speech embedding sum returned wrong extent".into());
        }
        let unit_weight = engine.htod(&[1.0f32; HIDDEN])?;
        for (index, layer) in self.local_layers.iter().enumerate() {
            let normalized = norm(engine, &hidden, &layer.input_norm, &unit_weight, rows)?;
            let mut query =
                linear_bias(engine, &layer.query, &layer.query_bias, &normalized, rows)?;
            let mut key = linear_bias(engine, &layer.key, &layer.key_bias, &normalized, rows)?;
            let value = linear_bias(engine, &layer.value, &layer.value_bias, &normalized, rows)?;
            rope(engine, &mut query, &mut key, groups)?;
            let attended = engine.mimo_audio_preprojected_attention(
                &self.plan, index, &query, &key, &value, groups,
            )?;
            let attended = round(engine, &attended, HIDDEN)?;
            let attention_output = linear(engine, &layer.attention_output, &attended, rows)?;
            let after_attention = epilogue(
                engine,
                &hidden,
                Some(&attention_output),
                HIDDEN,
                Epilogue::Residual,
            )?;
            let normalized = norm(
                engine,
                &after_attention,
                &layer.post_attention_norm,
                &unit_weight,
                rows,
            )?;
            let gate = linear(engine, &layer.mlp_gate, &normalized, rows)?;
            let up = linear(engine, &layer.mlp_up, &normalized, rows)?;
            let activated = epilogue(engine, &gate, Some(&up), FF, Epilogue::SiluMul)?;
            let mlp_output = linear(engine, &layer.mlp_down, &activated, rows)?;
            hidden = epilogue(
                engine,
                &after_attention,
                Some(&mlp_output),
                HIDDEN,
                Epilogue::Residual,
            )?;
        }
        let normalized = norm(engine, &hidden, &self.final_norm, &unit_weight, rows)?;
        // `[groups, 4, 1024]` is contiguous. Flatten its last two axes only
        // after the local transformer and its final norm.
        let projected = linear(engine, &self.projection_in, &normalized, groups)?;
        let activated = epilogue(engine, &projected, None, PROJECTION_FF, Epilogue::Gelu)?;
        let output = linear(engine, &self.projection_out, &activated, groups)?;
        ensure_finite(engine, &output)?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memra_gguf::config::{HfConfig, ModelConfig};
    use memra_gguf::model_packs::mimo_v2::audio::pinned_patch_plan;

    fn plan() -> MiMoAudioPatchPlan {
        let config = ModelConfig::from_hf(&HfConfig::parse(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../memra-gguf/src/model_packs/mimo_v2/fixtures/config.json"
        ))));
        pinned_patch_plan(&config).unwrap()
    }

    #[test]
    fn pinned_forward_exposes_source_shape_and_group_cap() {
        let source = plan();
        assert_eq!(pinned_forward_plan(&source, 1).unwrap(), 4);
        assert_eq!(pinned_forward_plan(&source, 1_500).unwrap(), 6_000);
        assert!(pinned_forward_plan(&source, 0).is_err());
        assert!(pinned_forward_plan(&source, 1_501).is_err());
        let mut changed = source;
        changed.local_full_attention = false;
        assert!(pinned_forward_plan(&changed, 1).is_err());
        let mut changed = source;
        changed.local_rope_theta += 1.0;
        assert!(pinned_forward_plan(&changed, 1).is_err());
        let mut changed = source;
        changed.projection_intermediate = 4_096;
        assert!(pinned_forward_plan(&changed, 1).is_err());
    }

    fn bf16(value: f32) -> f32 {
        let bits = value.to_bits();
        let rounding = 0x7fff + ((bits >> 16) & 1);
        f32::from_bits((bits.wrapping_add(rounding) >> 16) << 16)
    }

    #[test]
    fn bf16_source_boundaries_change_bias_silu_and_residual_values() {
        // Linear bias is added to the f32 accumulation before its BF16 store.
        let dot = 1.003f32;
        let bias = 0.003f32;
        assert_ne!(bf16(dot + bias), bf16(bf16(dot) + bias));
        let gate = bf16(1.25);
        let up = bf16(0.75);
        let silu = bf16(gate / (1.0 + (-gate).exp()));
        let product = bf16(silu * up);
        assert_eq!(product.to_bits(), bf16(product).to_bits());
        let residual = bf16(bf16(1.003) + bf16(0.003));
        assert_eq!(residual.to_bits(), bf16(residual).to_bits());
        // Qwen2RMSNorm casts the unit-normalized value to the input dtype
        // before the learned BF16 weight multiplication.
        assert_ne!(bf16(bf16(0.333) * bf16(0.777)), bf16(0.333 * bf16(0.777)));
    }

    #[test]
    fn rotary_positions_restart_for_each_four_code_group() {
        fn pair(first: f32, second: f32, token: usize, dimension: usize) -> (f32, f32) {
            let inverse_frequency =
                1.0 / 640_000.0f32.powf((2 * dimension) as f32 / HEAD_DIM as f32);
            let angle = (token % GROUP) as f32 * inverse_frequency;
            let cosine = bf16(angle.cos());
            let sine = bf16(angle.sin());
            (
                bf16(bf16(first * cosine) - bf16(second * sine)),
                bf16(bf16(second * cosine) + bf16(first * sine)),
            )
        }
        assert_eq!(pair(1.0, 2.0, 0, 0), pair(1.0, 2.0, 4, 0));
        assert_eq!(pair(1.0, 2.0, 3, 0), pair(1.0, 2.0, 7, 0));
        assert_ne!(pair(1.0, 2.0, 0, 0), pair(1.0, 2.0, 1, 0));
    }

    #[test]
    #[ignore = "requires a dedicated MiMo GPU component lane"]
    fn gpu_bf16_bias_rope_and_activation_match_source_order() -> Result<(), Fail> {
        let gpu: usize = std::env::var("MEMRA_MIMO_COMPONENT_GPU")
            .unwrap_or_else(|_| "0".into())
            .parse()?;
        let engine = Engine::new(gpu)?;
        let dot = engine.htod(&[1.003, -1.003, 0.5, -0.5])?;
        let bias = engine.htod(&[0.003, -0.003, 0.25, -0.25])?;
        let biased = epilogue(&engine, &dot, Some(&bias), 4, Epilogue::Bias)?;
        let actual = engine.dtoh(&biased)?;
        let expected = [1.003, -1.003, 0.5, -0.5]
            .into_iter()
            .zip([0.003, -0.003, 0.25, -0.25])
            .map(|(value, bias)| bf16(value + bias))
            .collect::<Vec<_>>();
        for (got, want) in actual.into_iter().zip(expected) {
            assert_eq!(got.to_bits(), want.to_bits());
        }

        let normalized = engine.htod(&[0.333, -0.333, 1.003, -1.003])?;
        let norm_weight = engine.htod(&[bf16(0.777), bf16(0.777), 1.0, 1.0])?;
        let normed = epilogue(
            &engine,
            &normalized,
            Some(&norm_weight),
            4,
            Epilogue::NormMul,
        )?;
        let actual = engine.dtoh(&normed)?;
        assert_eq!(
            actual[0].to_bits(),
            bf16(bf16(0.333) * bf16(0.777)).to_bits()
        );
        assert_eq!(
            actual[1].to_bits(),
            bf16(bf16(-0.333) * bf16(0.777)).to_bits()
        );

        let gate = engine.htod(&[0.5, 1.25, -0.75, 2.0])?;
        let up = engine.htod(&[1.0, 0.75, -1.0, 0.5])?;
        let product = epilogue(&engine, &gate, Some(&up), 4, Epilogue::SiluMul)?;
        let actual = engine.dtoh(&product)?;
        for ((got, gate), up) in actual
            .into_iter()
            .zip([0.5f32, 1.25, -0.75, 2.0])
            .zip([1.0f32, 0.75, -1.0, 0.5])
        {
            let want = bf16(bf16(gate / (1.0 + (-gate).exp())) * up);
            assert_eq!(got.to_bits(), want.to_bits());
        }

        let mut host_q = vec![0.0f32; 2 * GROUP * HIDDEN];
        let mut host_k = vec![0.0f32; 2 * GROUP * HIDDEN];
        for group in 0..2 {
            for token in 0..GROUP {
                let row = (group * GROUP + token) * HIDDEN;
                host_q[row] = 1.0;
                host_q[row + 32] = 2.0;
                host_k[row] = -1.0;
                host_k[row + 32] = 0.5;
            }
        }
        let mut query = engine.htod(&host_q)?;
        let mut key = engine.htod(&host_k)?;
        rope(&engine, &mut query, &mut key, 2)?;
        let q = engine.dtoh(&query)?;
        let k = engine.dtoh(&key)?;
        for token in 0..GROUP {
            let angle = token as f32;
            let cosine = bf16(angle.cos());
            let sine = bf16(angle.sin());
            let base0 = token * HIDDEN;
            let base1 = (token + GROUP) * HIDDEN;
            let want_q0 = bf16(bf16(cosine) - bf16(2.0 * sine));
            let want_q1 = bf16(bf16(2.0 * cosine) + bf16(sine));
            let want_k0 = bf16(bf16(-cosine) - bf16(0.5 * sine));
            let want_k1 = bf16(bf16(0.5 * cosine) + bf16(-sine));
            for base in [base0, base1] {
                assert_eq!(q[base].to_bits(), want_q0.to_bits());
                assert_eq!(q[base + 32].to_bits(), want_q1.to_bits());
                assert_eq!(k[base].to_bits(), want_k0.to_bits());
                assert_eq!(k[base + 32].to_bits(), want_k1.to_bits());
            }
        }
        Ok(())
    }
}
