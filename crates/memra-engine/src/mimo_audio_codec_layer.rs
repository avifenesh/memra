//! Source-backed, bounded one-layer MiMo bundled audio-tokenizer encoder.
//!
//! This consumes the resident BF16 rows. It does not run the other 23 layers,
//! encoder skip/pool, RVQ, PCM processing, or a serving request.

use core::ffi::c_void;
use std::error::Error;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use memra_reference::mimo_audio_codec_layer::MAX_COMPONENT_TOKENS;

use crate::Engine;
use crate::mimo_audio_codec_weights::{
    CodecEncoderBf16Linear, CodecEncoderBf16Norm, MiMoAudioCodecEncoderWeights,
};

type Fail = Box<dyn Error>;
const HIDDEN: usize = 1_024;

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
        || layer_index >= 24
    {
        return Err("MiMo codec one-layer token count, input extent, or layer changed".into());
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
        let residual = epilogue(
            engine,
            &projection,
            None,
            Some(input),
            HIDDEN,
            Epilogue::Residual,
        )?;
        let mlp_input = normalized(engine, &residual, &weights.final_norm, tokens)?;
        let hidden = projected(engine, &mlp_input, &weights.fc1, tokens)?;
        let hidden = epilogue(engine, &hidden, None, None, 4_096, Epilogue::Gelu)?;
        let projection = projected(engine, &hidden, &weights.fc2, tokens)?;
        let output = epilogue(
            engine,
            &projection,
            None,
            Some(&residual),
            HIDDEN,
            Epilogue::Residual,
        )?;
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
}
