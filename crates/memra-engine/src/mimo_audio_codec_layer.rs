//! Source-backed, bounded MiMo bundled audio-tokenizer transformer encoder.
//!
//! This consumes the resident BF16 rows after the convolutional frontend. It
//! includes the source post-stack downsample. RVQ and PCM remain separate.

use core::ffi::c_void;
use std::error::Error;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use memra_gguf::model_packs::mimo_v2::audio_tokenizer::AudioTokenizerAuxiliaryContract;
use memra_reference::mimo_audio_codec_layer::{
    ENCODER_LAYERS, ENCODER_SKIP_LAYER_ID, MAX_COMPONENT_TOKENS,
};

use crate::Engine;
use crate::mimo_audio_codec_weights::{
    CodecEncoderBf16Linear, CodecEncoderBf16Norm, MiMoAudioCodecEncoderWeights,
};

type Fail = Box<dyn Error>;
const HIDDEN: usize = 1_024;

#[cfg(test)]
std::thread_local! {
    static LAYER0_STAGES: std::cell::RefCell<Option<Vec<(&'static str, Vec<f32>)>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(crate) struct Layer0StageCaptureGuard;

#[cfg(test)]
pub(crate) fn start_layer0_stage_capture() -> Layer0StageCaptureGuard {
    LAYER0_STAGES.with(|capture| {
        assert!(
            capture.borrow_mut().replace(Vec::new()).is_none(),
            "MiMo codec layer-0 capture is already active"
        );
    });
    Layer0StageCaptureGuard
}

#[cfg(test)]
impl Layer0StageCaptureGuard {
    pub(crate) fn finish(self) -> Vec<(&'static str, Vec<f32>)> {
        LAYER0_STAGES.with(|capture| capture.borrow_mut().take().unwrap_or_default())
    }
}

#[cfg(test)]
impl Drop for Layer0StageCaptureGuard {
    fn drop(&mut self) {
        LAYER0_STAGES.with(|capture| {
            capture.borrow_mut().take();
        });
    }
}

#[cfg(test)]
fn capture_layer0_stage(
    engine: &Engine,
    layer_index: usize,
    label: &'static str,
    values: &CudaSlice<f32>,
) -> Result<(), Fail> {
    if layer_index == 0 && LAYER0_STAGES.with(|capture| capture.borrow().is_some()) {
        let host = engine.dtoh(values)?;
        LAYER0_STAGES.with(|capture| {
            capture
                .borrow_mut()
                .as_mut()
                .expect("MiMo codec layer-0 capture disappeared")
                .push((label, host));
        });
    }
    Ok(())
}

unsafe extern "C" {
    fn memra_mimo_codec_layer_check_values(
        values: *const f32,
        elements: i32,
        require_bf16: i32,
        fault: *mut i32,
        stream: *mut c_void,
    ) -> i32;
    fn memra_mimo_codec_layer_norm(
        input: *const f32,
        weight: *const u16,
        bias: *const u16,
        output: *mut f32,
        tokens: i32,
        stream: *mut c_void,
    ) -> i32;
    fn memra_mimo_codec_layer_linear_direct(
        input: *const f32,
        weight: *const u16,
        output: *mut f32,
        rows: i32,
        input_features: i32,
        output_features: i32,
        stream: *mut c_void,
    ) -> i32;
    fn memra_mimo_codec_downsample_columns(
        input: *const f32,
        columns: *mut f32,
        tokens: i32,
        stream: *mut c_void,
    ) -> i32;
    fn memra_mimo_codec_layer_epilogue(
        input: *const f32,
        bias: *const u16,
        residual: *const f32,
        output: *mut f32,
        elements: i32,
        width: i32,
        operation: i32,
        stream: *mut c_void,
    ) -> i32;
    fn memra_mimo_codec_layer_rope(
        query: *mut f32,
        key: *mut f32,
        tokens: i32,
        stream: *mut c_void,
    ) -> i32;
    fn memra_mimo_codec_layer_attention(
        query: *const f32,
        key: *const f32,
        value: *const f32,
        output: *mut f32,
        tokens: i32,
        window: i32,
        stream: *mut c_void,
    ) -> i32;
}

#[derive(Clone, Copy)]
#[repr(i32)]
enum Epilogue {
    Round = 0,
    Bias = 1,
    Residual = 2,
    Gelu = 3,
}

fn request(tokens: usize, elements: usize, layer_index: usize) -> Result<(), Fail> {
    if !(1..=MAX_COMPONENT_TOKENS).contains(&tokens)
        || elements != tokens * HIDDEN
        || layer_index >= ENCODER_LAYERS
    {
        return Err("MiMo codec one-layer token count, input extent, or layer changed".into());
    }
    Ok(())
}

fn stack_plan(contract: &AudioTokenizerAuxiliaryContract) -> Result<(), Fail> {
    // The resident loader verifies the exact pinned config, including
    // encoder_skip_layer_id=3, LayerNorm, causal attention, and avg_pooler=2.
    if contract.encoder_layers != ENCODER_LAYERS
        || contract.hidden_size != HIDDEN
        || contract.attention_heads != 16
        || contract.ffn_size != 4_096
        || !contract.hybrid_attention
        || contract.hybrid_block_size != 8
        || contract.swa_per_block != 2
        || ENCODER_SKIP_LAYER_ID != 3
    {
        return Err("MiMo codec encoder source stack plan changed".into());
    }
    Ok(())
}

fn checked_rc(op: &str, rc: i32) -> Result<(), Fail> {
    if rc != 0 {
        return Err(format!("MiMo codec {op} CUDA refusal {rc}").into());
    }
    Ok(())
}

fn check_input(engine: &Engine, input: &CudaSlice<f32>) -> Result<(), Fail> {
    let stream = engine.stream();
    let mut fault = engine.htod_i32(&[0])?;
    let (input_ptr, input_guard) = input.device_ptr(&stream);
    let (fault_ptr, fault_guard) = fault.device_ptr_mut(&stream);
    let rc = unsafe {
        memra_mimo_codec_layer_check_values(
            input_ptr as *const f32,
            input.len() as i32,
            1,
            fault_ptr as *mut i32,
            stream.cu_stream() as *mut c_void,
        )
    };
    drop((input_guard, fault_guard));
    checked_rc("input finite check", rc)?;
    if engine.dtoh_i32(&fault)? != [0] {
        return Err("MiMo codec one-layer input is non-finite or not BF16-valued".into());
    }
    Ok(())
}

fn check_result(engine: &Engine, result: &CudaSlice<f32>) -> Result<(), Fail> {
    let stream = engine.stream();
    let mut fault = engine.htod_i32(&[0])?;
    let (result_ptr, result_guard) = result.device_ptr(&stream);
    let (fault_ptr, fault_guard) = fault.device_ptr_mut(&stream);
    let rc = unsafe {
        memra_mimo_codec_layer_check_values(
            result_ptr as *const f32,
            result.len() as i32,
            1,
            fault_ptr as *mut i32,
            stream.cu_stream() as *mut c_void,
        )
    };
    drop((result_guard, fault_guard));
    checked_rc("output finite check", rc)?;
    if engine.dtoh_i32(&fault)? != [0] {
        return Err("MiMo codec one-layer output is non-finite or not BF16-valued".into());
    }
    Ok(())
}

fn normalized(
    engine: &Engine,
    input: &CudaSlice<f32>,
    norm: &CodecEncoderBf16Norm<'_>,
    tokens: usize,
) -> Result<CudaSlice<f32>, Fail> {
    let stream = engine.stream();
    let mut output = engine.uninit(input.len())?;
    let (input_ptr, input_guard) = input.device_ptr(&stream);
    let (weight_ptr, weight_guard) = norm.weight.device_ptr(&stream);
    let (bias_ptr, bias_guard) = norm.bias.device_ptr(&stream);
    let (output_ptr, output_guard) = output.device_ptr_mut(&stream);
    let rc = unsafe {
        memra_mimo_codec_layer_norm(
            input_ptr as *const f32,
            weight_ptr as *const u16,
            bias_ptr as *const u16,
            output_ptr as *mut f32,
            tokens as i32,
            stream.cu_stream() as *mut c_void,
        )
    };
    drop((input_guard, weight_guard, bias_guard, output_guard));
    checked_rc("LayerNorm", rc)?;
    Ok(output)
}

fn epilogue(
    engine: &Engine,
    input: &CudaSlice<f32>,
    bias: Option<&CudaSlice<u8>>,
    residual: Option<&CudaSlice<f32>>,
    width: usize,
    operation: Epilogue,
) -> Result<CudaSlice<f32>, Fail> {
    let stream = engine.stream();
    let ordinal = stream.context().ordinal();
    if width == 0
        || input.is_empty()
        || !input.len().is_multiple_of(width)
        || input.len() > MAX_COMPONENT_TOKENS * 4_096
        || input.ordinal() != ordinal
        || bias.is_some_and(|rhs| rhs.len() != width * 2 || rhs.ordinal() != ordinal)
        || residual.is_some_and(|rhs| rhs.len() != input.len() || rhs.ordinal() != ordinal)
        || matches!(operation, Epilogue::Bias) != bias.is_some()
        || matches!(operation, Epilogue::Residual) != residual.is_some()
    {
        return Err("MiMo codec BF16 epilogue extent or GPU changed".into());
    }
    let mut output = engine.uninit(input.len())?;
    let (input_ptr, input_guard) = input.device_ptr(&stream);
    let (bias_ptr, bias_guard) = if let Some(bias) = bias {
        let (ptr, guard) = bias.device_ptr(&stream);
        (ptr as *const u16, Some(guard))
    } else {
        (std::ptr::null(), None)
    };
    let (residual_ptr, residual_guard) = if let Some(residual) = residual {
        let (ptr, guard) = residual.device_ptr(&stream);
        (ptr as *const f32, Some(guard))
    } else {
        (std::ptr::null(), None)
    };
    let (output_ptr, output_guard) = output.device_ptr_mut(&stream);
    let rc = unsafe {
        memra_mimo_codec_layer_epilogue(
            input_ptr as *const f32,
            bias_ptr,
            residual_ptr,
            output_ptr as *mut f32,
            input.len() as i32,
            width as i32,
            operation as i32,
            stream.cu_stream() as *mut c_void,
        )
    };
    drop((input_guard, bias_guard, residual_guard, output_guard));
    checked_rc("BF16 epilogue", rc)?;
    Ok(output)
}

fn projected(
    engine: &Engine,
    input: &CudaSlice<f32>,
    op: &CodecEncoderBf16Linear<'_>,
    tokens: usize,
) -> Result<CudaSlice<f32>, Fail> {
    let stream = engine.stream();
    let ordinal = stream.context().ordinal();
    if input.len() != tokens * op.input
        || op.weight.len() != op.input * op.output * 2
        || input.ordinal() != ordinal
        || op.weight.ordinal() != ordinal
        || op
            .bias
            .is_some_and(|bias| bias.ordinal() != ordinal || bias.len() != op.output * 2)
    {
        return Err("MiMo codec projection shape or GPU changed".into());
    }
    let unrounded = match engine.bf16_tc_gemm(op.weight, input, tokens, op.input, op.output)? {
        Some(projected) => projected,
        None => {
            let mut projected = engine.uninit(tokens * op.output)?;
            let (input_ptr, input_guard) = input.device_ptr(&stream);
            let (weight_ptr, weight_guard) = op.weight.device_ptr(&stream);
            let (output_ptr, output_guard) = projected.device_ptr_mut(&stream);
            let rc = unsafe {
                memra_mimo_codec_layer_linear_direct(
                    input_ptr as *const f32,
                    weight_ptr as *const u16,
                    output_ptr as *mut f32,
                    tokens as i32,
                    op.input as i32,
                    op.output as i32,
                    stream.cu_stream() as *mut c_void,
                )
            };
            drop((input_guard, weight_guard, output_guard));
            checked_rc("direct BF16 projection", rc)?;
            projected
        }
    };
    epilogue(
        engine,
        &unrounded,
        op.bias,
        None,
        op.output,
        if op.bias.is_some() {
            Epilogue::Bias
        } else {
            Epilogue::Round
        },
    )
}

fn downsample_columns(
    engine: &Engine,
    input: &CudaSlice<f32>,
    tokens: usize,
) -> Result<CudaSlice<f32>, Fail> {
    let output_tokens = tokens.div_ceil(2);
    let mut columns = engine.uninit(output_tokens * HIDDEN * 2)?;
    let stream = engine.stream();
    let (input_ptr, input_guard) = input.device_ptr(&stream);
    let (columns_ptr, columns_guard) = columns.device_ptr_mut(&stream);
    let rc = unsafe {
        memra_mimo_codec_downsample_columns(
            input_ptr as *const f32,
            columns_ptr as *mut f32,
            tokens as i32,
            stream.cu_stream() as *mut c_void,
        )
    };
    drop((input_guard, columns_guard));
    checked_rc("stride-two Conv1D columns", rc)?;
    Ok(columns)
}

fn rotary(
    engine: &Engine,
    query: &mut CudaSlice<f32>,
    key: &mut CudaSlice<f32>,
    tokens: usize,
) -> Result<(), Fail> {
    let stream = engine.stream();
    let (query_ptr, query_guard) = query.device_ptr_mut(&stream);
    let (key_ptr, key_guard) = key.device_ptr_mut(&stream);
    let rc = unsafe {
        memra_mimo_codec_layer_rope(
            query_ptr as *mut f32,
            key_ptr as *mut f32,
            tokens as i32,
            stream.cu_stream() as *mut c_void,
        )
    };
    drop((query_guard, key_guard));
    checked_rc("theta-10000 RoPE", rc)
}

fn attended(
    engine: &Engine,
    query: &CudaSlice<f32>,
    key: &CudaSlice<f32>,
    value: &CudaSlice<f32>,
    tokens: usize,
    window: Option<usize>,
) -> Result<CudaSlice<f32>, Fail> {
    let stream = engine.stream();
    let ordinal = stream.context().ordinal();
    if [query, key, value]
        .iter()
        .any(|row| row.len() != tokens * HIDDEN || row.ordinal() != ordinal)
    {
        return Err("MiMo codec attention QKV shape or GPU changed".into());
    }
    let mut output = engine.uninit(tokens * HIDDEN)?;
    let (query_ptr, query_guard) = query.device_ptr(&stream);
    let (key_ptr, key_guard) = key.device_ptr(&stream);
    let (value_ptr, value_guard) = value.device_ptr(&stream);
    let (output_ptr, output_guard) = output.device_ptr_mut(&stream);
    let rc = unsafe {
        memra_mimo_codec_layer_attention(
            query_ptr as *const f32,
            key_ptr as *const f32,
            value_ptr as *const f32,
            output_ptr as *mut f32,
            tokens as i32,
            window.map_or(-1, |size| size as i32),
            stream.cu_stream() as *mut c_void,
        )
    };
    drop((query_guard, key_guard, value_guard, output_guard));
    checked_rc("causal attention", rc)?;
    Ok(output)
}

impl MiMoAudioCodecEncoderWeights {
    /// Execute exactly one selected transformer layer over one frontended,
    /// BF16-valued `[tokens,1024]` sequence on the weight-owning GPU.
    ///
    /// This is a component path capped at 256 tokens. It applies the pinned
    /// alternating 128-prior-token/full causal mask and source BF16 boundaries.
    pub fn encode_one_transformer_layer(
        &self,
        engine: &Engine,
        input: &CudaSlice<f32>,
        tokens: usize,
        layer_index: usize,
    ) -> Result<CudaSlice<f32>, Fail> {
        request(tokens, input.len(), layer_index)?;
        engine.gpu.ctx.bind_to_thread()?;
        self.check_device(engine)?;
        if input.ordinal() != self.device_ordinal {
            return Err("MiMo codec input belongs to another GPU".into());
        }
        let weights = self.encoder_layer_bf16(layer_index)?;
        check_input(engine, input)?;
        let attention_input = normalized(engine, input, &weights.attention_norm, tokens)?;
        #[cfg(test)]
        capture_layer0_stage(engine, layer_index, "layer0_attn_norm", &attention_input)?;
        let mut query = projected(engine, &attention_input, &weights.query, tokens)?;
        let mut key = projected(engine, &attention_input, &weights.key, tokens)?;
        let value = projected(engine, &attention_input, &weights.value, tokens)?;
        rotary(engine, &mut query, &mut key, tokens)?;
        let context = attended(
            engine,
            &query,
            &key,
            &value,
            tokens,
            weights.attention_window(),
        )?;
        let projection = projected(engine, &context, &weights.attention_output, tokens)?;
        #[cfg(test)]
        capture_layer0_stage(engine, layer_index, "layer0_attention", &projection)?;
        let residual = epilogue(
            engine,
            &projection,
            None,
            Some(input),
            HIDDEN,
            Epilogue::Residual,
        )?;
        let mlp_input = normalized(engine, &residual, &weights.final_norm, tokens)?;
        #[cfg(test)]
        capture_layer0_stage(engine, layer_index, "layer0_mlp_norm", &mlp_input)?;
        let hidden = projected(engine, &mlp_input, &weights.fc1, tokens)?;
        let hidden = epilogue(engine, &hidden, None, None, 4_096, Epilogue::Gelu)?;
        let projection = projected(engine, &hidden, &weights.fc2, tokens)?;
        #[cfg(test)]
        capture_layer0_stage(engine, layer_index, "layer0_fc2", &projection)?;
        let output = epilogue(
            engine,
            &projection,
            None,
            Some(&residual),
            HIDDEN,
            Epilogue::Residual,
        )?;
        #[cfg(test)]
        capture_layer0_stage(engine, layer_index, "layer0", &output)?;
        check_result(engine, &output)?;
        Ok(output)
    }

    /// Execute source-order layers 0..23 on already-frontended BF16-valued
    /// `[tokens,1024]`, add the output after layer index 2, and apply the
    /// source `encoder.layer_norm`. This is a 1..256 token component; it does
    /// not downsample, quantize, process PCM, or handle full raw audio.
    pub fn encode_transformer_stack(
        &self,
        engine: &Engine,
        input: &CudaSlice<f32>,
        tokens: usize,
    ) -> Result<CudaSlice<f32>, Fail> {
        request(tokens, input.len(), 0)?;
        stack_plan(&self.contract)?;
        engine.gpu.ctx.bind_to_thread()?;
        self.check_device(engine)?;
        if input.ordinal() != self.device_ordinal {
            return Err("MiMo codec input belongs to another GPU".into());
        }
        let final_norm = self.encoder_final_norm_bf16()?;
        // Refuse a partial stack before launching any layer.
        for index in 0..ENCODER_LAYERS {
            self.encoder_layer_bf16(index)?;
        }

        let mut hidden = self.encode_one_transformer_layer(engine, input, tokens, 0)?;
        for index in 1..ENCODER_SKIP_LAYER_ID {
            hidden = self.encode_one_transformer_layer(engine, &hidden, tokens, index)?;
        }
        let skip = hidden;
        let mut hidden =
            self.encode_one_transformer_layer(engine, &skip, tokens, ENCODER_SKIP_LAYER_ID)?;
        for index in ENCODER_SKIP_LAYER_ID + 1..ENCODER_LAYERS {
            hidden = self.encode_one_transformer_layer(engine, &hidden, tokens, index)?;
        }
        let hidden = epilogue(
            engine,
            &hidden,
            None,
            Some(&skip),
            HIDDEN,
            Epilogue::Residual,
        )?;
        let output = normalized(engine, &hidden, &final_norm, tokens)?;
        check_result(engine, &output)?;
        Ok(output)
    }

    /// Downsample one already-frontended and 24-layer-encoded BF16-valued
    /// `[tokens,1024]` sequence after `encode_transformer_stack`. The source
    /// pads an odd final token with zeros, applies its biasless BF16 Conv1D
    /// `[1024,1024,2]` with stride two, erf GELU, then BF16 affine LayerNorm.
    /// The output carries `ceil(tokens/2)` BF16-valued rows in f32 slots.
    ///
    /// This bounded component stops before RVQ and accepts 1..=256 tokens.
    pub fn downsample_post_stack(
        &self,
        engine: &Engine,
        input: &CudaSlice<f32>,
        tokens: usize,
    ) -> Result<CudaSlice<f32>, Fail> {
        request(tokens, input.len(), 0)?;
        stack_plan(&self.contract)?;
        engine.gpu.ctx.bind_to_thread()?;
        self.check_device(engine)?;
        if input.ordinal() != self.device_ordinal {
            return Err("MiMo codec input belongs to another GPU".into());
        }
        let downsample = self.encoder_downsample_bf16()?;
        check_input(engine, input)?;
        let output_tokens = tokens.div_ceil(2);
        let columns = downsample_columns(engine, input, tokens)?;
        let convolved = projected(engine, &columns, &downsample.projection, output_tokens)?;
        let activated = epilogue(engine, &convolved, None, None, HIDDEN, Epilogue::Gelu)?;
        let output = normalized(engine, &activated, &downsample.norm, output_tokens)?;
        check_result(engine, &output)?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_layer_request_refuses_missing_or_excess_tokens() {
        assert!(request(1, HIDDEN, 0).is_ok());
        assert!(request(256, 256 * HIDDEN, 23).is_ok());
        assert!(request(0, 0, 0).is_err());
        assert!(request(257, 257 * HIDDEN, 0).is_err());
        assert!(request(1, HIDDEN - 1, 0).is_err());
        assert!(request(1, HIDDEN, 24).is_err());
        assert_eq!(1usize.div_ceil(2), 1);
        assert_eq!(255usize.div_ceil(2), 128);
        assert_eq!(256usize.div_ceil(2), 128);
    }

    #[test]
    fn source_stack_plan_requires_pinned_encoder_geometry() {
        use memra_gguf::model_packs::mimo_v2::audio_tokenizer::{
            LFS_WEIGHT_SHA256, SOURCE, verify_pinned_auxiliary,
        };

        let config = include_bytes!(
            "../../memra-gguf/src/model_packs/mimo_v2/fixtures/audio-tokenizer-config.json"
        );
        let header = include_bytes!(
            "../../memra-gguf/src/model_packs/mimo_v2/fixtures/audio-tokenizer-header.json"
        );
        let mut contract =
            verify_pinned_auxiliary(SOURCE, config, header, LFS_WEIGHT_SHA256, 1_872_618_384)
                .unwrap();
        stack_plan(&contract).unwrap();
        contract.encoder_layers = 23;
        assert!(stack_plan(&contract).is_err());
        contract.encoder_layers = 24;
        contract.swa_per_block = 1;
        assert!(stack_plan(&contract).is_err());
    }

    #[test]
    #[ignore = "requires the pinned bundled weight file and a dedicated GPU qualification"]
    fn pinned_one_layer_gpu_parity_gate() -> Result<(), Fail> {
        use memra_reference::mimo_audio_codec_layer::{Layer, Linear, Norm, forward};
        use std::collections::BTreeMap;
        use std::path::Path;

        let root = std::env::var("MIMO_PINNED_SOURCE_ROOT")?;
        let gpu: usize = std::env::var("MEMRA_MIMO_COMPONENT_GPU")
            .unwrap_or_else(|_| "0".into())
            .parse()?;
        let engine = Engine::new(gpu)?;
        let resident = MiMoAudioCodecEncoderWeights::load(&engine, Path::new(&root))?;
        let mut host = BTreeMap::new();
        for (name, tensor) in resident.tensors() {
            if !name.starts_with("encoder.layers.0.") {
                continue;
            }
            let bytes = engine.dtoh_u8(&tensor.bytes)?;
            host.insert(
                name.trim_start_matches("encoder.layers.0.").to_string(),
                bytes
                    .chunks_exact(2)
                    .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect::<Vec<_>>(),
            );
        }
        let fetch = |name: &str| host.get(name).map(Vec::as_slice).unwrap();
        let norm = |stem: &str| Norm {
            weight: fetch(&format!("{stem}.weight")),
            bias: fetch(&format!("{stem}.bias")),
        };
        let linear = |stem: &str, input: usize, output: usize, bias: bool| Linear {
            weight: fetch(&format!("{stem}.weight")),
            bias: bias.then(|| fetch(&format!("{stem}.bias"))),
            input,
            output,
        };
        let layer = Layer {
            attention_norm: norm("self_attn_layer_norm"),
            query: linear("self_attn.q_proj", HIDDEN, HIDDEN, true),
            key: linear("self_attn.k_proj", HIDDEN, HIDDEN, false),
            value: linear("self_attn.v_proj", HIDDEN, HIDDEN, true),
            attention_output: linear("self_attn.out_proj", HIDDEN, HIDDEN, true),
            final_norm: norm("final_layer_norm"),
            fc1: linear("fc1", HIDDEN, 4_096, true),
            fc2: linear("fc2", 4_096, HIDDEN, true),
            heads: 16,
            window: Some(128),
        };
        let tokens = 2;
        let input = (0..tokens * HIDDEN)
            .map(|index| {
                memra_reference::mimo_audio_codec_layer::bf16(
                    (index as f32 % 17.0 - 8.0) * 0.015625,
                )
            })
            .collect::<Vec<_>>();
        let expected = forward(&input, tokens, &layer)?;
        let actual =
            resident.encode_one_transformer_layer(&engine, &engine.htod(&input)?, tokens, 0)?;
        for (index, (got, want)) in engine.dtoh(&actual)?.iter().zip(expected).enumerate() {
            assert!(
                (got - want).abs() <= 0.0625,
                "element {index}: {got} != {want}"
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires pinned bundled weights and a dedicated GPU for the 24-layer source gate"]
    fn pinned_encoder_stack_gpu_source_gate() -> Result<(), Fail> {
        use memra_reference::mimo_audio_codec_layer::{
            Layer, Linear, Norm, bf16, forward_encoder_stack,
        };
        use std::collections::BTreeMap;
        use std::path::Path;

        let root = std::env::var("MIMO_PINNED_SOURCE_ROOT")?;
        let gpu: usize = std::env::var("MEMRA_MIMO_COMPONENT_GPU")
            .unwrap_or_else(|_| "0".into())
            .parse()?;
        let engine = Engine::new(gpu)?;
        let resident = MiMoAudioCodecEncoderWeights::load(&engine, Path::new(&root))?;
        let mut host = BTreeMap::new();
        for (name, tensor) in resident.tensors() {
            if !name.starts_with("encoder.layers.") && !name.starts_with("encoder.layer_norm.") {
                continue;
            }
            let bytes = engine.dtoh_u8(&tensor.bytes)?;
            host.insert(
                name.to_string(),
                bytes
                    .chunks_exact(2)
                    .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect::<Vec<_>>(),
            );
        }
        assert_eq!(host.len(), ENCODER_LAYERS * 15 + 2);
        let fetch = |name: &str| host.get(name).map(Vec::as_slice).unwrap();
        let norm = |stem: &str| Norm {
            weight: fetch(&format!("{stem}.weight")),
            bias: fetch(&format!("{stem}.bias")),
        };
        let linear = |stem: &str, input: usize, output: usize, bias: bool| Linear {
            weight: fetch(&format!("{stem}.weight")),
            bias: bias.then(|| fetch(&format!("{stem}.bias"))),
            input,
            output,
        };
        let layers = (0..ENCODER_LAYERS)
            .map(|index| {
                let stem = format!("encoder.layers.{index}");
                let attention = format!("{stem}.self_attn");
                Layer {
                    attention_norm: norm(&format!("{stem}.self_attn_layer_norm")),
                    query: linear(&format!("{attention}.q_proj"), HIDDEN, HIDDEN, true),
                    key: linear(&format!("{attention}.k_proj"), HIDDEN, HIDDEN, false),
                    value: linear(&format!("{attention}.v_proj"), HIDDEN, HIDDEN, true),
                    attention_output: linear(
                        &format!("{attention}.out_proj"),
                        HIDDEN,
                        HIDDEN,
                        true,
                    ),
                    final_norm: norm(&format!("{stem}.final_layer_norm")),
                    fc1: linear(&format!("{stem}.fc1"), HIDDEN, 4_096, true),
                    fc2: linear(&format!("{stem}.fc2"), 4_096, HIDDEN, true),
                    heads: 16,
                    window: (index % 2 == 0).then_some(128),
                }
            })
            .collect::<Vec<_>>();
        let final_norm = norm("encoder.layer_norm");
        let tokens = 2;
        let input = (0..tokens * HIDDEN)
            .map(|index| bf16((index as f32 % 17.0 - 8.0) * 0.015625))
            .collect::<Vec<_>>();
        let expected = forward_encoder_stack(&input, tokens, &layers, &final_norm)?;
        let actual = resident.encode_transformer_stack(&engine, &engine.htod(&input)?, tokens)?;
        let actual = engine.dtoh(&actual)?;
        assert_eq!(actual.len(), expected.len());
        let max_abs = actual
            .iter()
            .zip(&expected)
            .map(|(got, want)| (got - want).abs())
            .fold(0.0f32, f32::max);
        let squared_error = actual
            .iter()
            .zip(&expected)
            .map(|(got, want)| f64::from(got - want).powi(2))
            .sum::<f64>();
        let reference_squared = expected
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>();
        let relative_l2 = (squared_error / reference_squared).sqrt();
        let bit_equal = actual
            .iter()
            .zip(&expected)
            .filter(|(got, want)| got.to_bits() == want.to_bits())
            .count();
        eprintln!(
            "MiMo codec 24-layer GPU {gpu}: max_abs={max_abs}, relative_l2={relative_l2}, bit_equal={bit_equal}/{}",
            actual.len()
        );
        for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (got - want).abs() <= 0.25,
                "24-layer encoder element {index}: {got} != {want}"
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires pinned bundled weights and a dedicated GPU for the post-stack source gate"]
    fn pinned_post_stack_downsample_gpu_source_gate() -> Result<(), Fail> {
        use memra_reference::mimo_audio_codec_layer::{Norm, bf16, post_stack_downsample};
        use std::path::Path;

        let root = std::env::var("MIMO_PINNED_SOURCE_ROOT")?;
        let gpu: usize = std::env::var("MEMRA_MIMO_COMPONENT_GPU")
            .unwrap_or_else(|_| "0".into())
            .parse()?;
        let engine = Engine::new(gpu)?;
        let resident = MiMoAudioCodecEncoderWeights::load(&engine, Path::new(&root))?;
        let fetch = |name: &str| -> Result<Vec<u16>, Fail> {
            let tensor = resident
                .tensor(name)
                .ok_or_else(|| format!("missing {name}"))?;
            let bytes = engine.dtoh_u8(&tensor.bytes)?;
            Ok(bytes
                .chunks_exact(2)
                .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
                .collect())
        };
        let weight = fetch("encoder.down_sample_layer.0.weight")?;
        let norm_weight = fetch("encoder.down_sample_norm.weight")?;
        let norm_bias = fetch("encoder.down_sample_norm.bias")?;
        let norm = Norm {
            weight: &norm_weight,
            bias: &norm_bias,
        };
        for tokens in [3, 4] {
            let input = (0..tokens * HIDDEN)
                .map(|index| bf16((index as f32 % 37.0 - 18.0) * 0.015625))
                .collect::<Vec<_>>();
            let expected = post_stack_downsample(&input, tokens, HIDDEN, &weight, &norm)?;
            let actual = resident.downsample_post_stack(&engine, &engine.htod(&input)?, tokens)?;
            let actual = engine.dtoh(&actual)?;
            assert_eq!(actual.len(), tokens.div_ceil(2) * HIDDEN);
            let max_abs = actual
                .iter()
                .zip(&expected)
                .map(|(got, want)| (got - want).abs())
                .fold(0.0f32, f32::max);
            let squared_error = actual
                .iter()
                .zip(&expected)
                .map(|(got, want)| f64::from(got - want).powi(2))
                .sum::<f64>();
            let reference_squared = expected
                .iter()
                .map(|value| f64::from(*value).powi(2))
                .sum::<f64>();
            let relative_l2 = (squared_error / reference_squared).sqrt();
            let bit_equal = actual
                .iter()
                .zip(&expected)
                .filter(|(got, want)| got.to_bits() == want.to_bits())
                .count();
            eprintln!(
                "MiMo codec downsample GPU {gpu}, tokens {tokens}: max_abs={max_abs}, relative_l2={relative_l2}, bit_equal={bit_equal}/{}",
                actual.len()
            );
            for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
                assert!(got.is_finite() && got.to_bits() & 0xffff == 0);
                assert!(
                    (got - want).abs() <= 0.0625,
                    "{tokens} tokens, element {index}: {got} != {want}"
                );
            }
        }
        Ok(())
    }
}
