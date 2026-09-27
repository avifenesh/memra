//! Pinned bundled audio-tokenizer encoder Conv1D frontend from prepared mel.
//!
//! This is only conv1, GELU, conv2, GELU. The 24 encoder layers, RVQ, PCM to
//! mel preprocessing, audio patch insertion, and serving are separate work.

use core::ffi::c_void;
use std::error::Error;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use memra_reference::mimo_audio_codec_frontend::output_frames;

use crate::Engine;
use crate::mimo_audio_codec_weights::MiMoAudioCodecEncoderWeights;

type Fail = Box<dyn Error>;

const MEL_CHANNELS: usize = 128;
const HIDDEN: usize = 1_024;

unsafe extern "C" {
    fn memra_mimo_codec_im2col_bf16(
        input: *const f32,
        columns: *mut f32,
        frames: i32,
        channels: i32,
        stride: i32,
        rows: i32,
        stream: *mut c_void,
    ) -> i32;
    fn memra_mimo_codec_conv_epilogue(
        projected: *const f32,
        bias: *const u16,
        output: *mut f32,
        rows: i32,
        out_channels: i32,
        stream: *mut c_void,
    ) -> i32;
    fn memra_mimo_codec_conv_direct(
        input: *const f32,
        weight: *const u16,
        bias: *const u16,
        output: *mut f32,
        frames: i32,
        in_channels: i32,
        out_channels: i32,
        stride: i32,
        rows: i32,
        stream: *mut c_void,
    ) -> i32;
    fn memra_mimo_codec_check_finite(
        values: *const f32,
        elements: i32,
        fault: *mut i32,
        stream: *mut c_void,
    ) -> i32;
}

#[derive(Clone, Copy)]
struct ConvSpec {
    frames: usize,
    in_channels: usize,
    out_channels: usize,
    stride: usize,
}

impl ConvSpec {
    fn validate(
        self,
        input_len: usize,
        weight_bytes: usize,
        bias_bytes: usize,
    ) -> Result<usize, &'static str> {
        let rows = output_frames(self.frames, self.stride)
            .map_err(|_| "MiMo codec Conv1D frame count or stride changed")?;
        if self.in_channels == 0
            || self.in_channels > HIDDEN
            || self.out_channels == 0
            || self.out_channels > HIDDEN
            || self.frames.checked_mul(self.in_channels) != Some(input_len)
            || self
                .out_channels
                .checked_mul(self.in_channels)
                .and_then(|extent| extent.checked_mul(6))
                != Some(weight_bytes)
            || self.out_channels.checked_mul(2) != Some(bias_bytes)
        {
            return Err("MiMo codec Conv1D input or BF16 weight extent changed");
        }
        Ok(rows)
    }
}

fn ensure_finite(engine: &Engine, values: &CudaSlice<f32>) -> Result<(), Fail> {
    engine.gpu.ctx.bind_to_thread()?;
    let stream = engine.stream();
    if values.is_empty()
        || values.len() > 1024 * 1024
        || values.ordinal() != stream.context().ordinal()
    {
        return Err("MiMo codec Conv1D finite-check extent or GPU changed".into());
    }
    let mut fault = engine.htod_i32(&[0])?;
    let (value_ptr, value_guard) = values.device_ptr(&stream);
    let (fault_ptr, fault_guard) = fault.device_ptr_mut(&stream);
    let rc = unsafe {
        memra_mimo_codec_check_finite(
            value_ptr as *const f32,
            values.len() as i32,
            fault_ptr as *mut i32,
            stream.cu_stream() as *mut c_void,
        )
    };
    drop((value_guard, fault_guard));
    if rc != 0 || engine.dtoh_i32(&fault)? != [0] {
        return Err(format!("MiMo codec Conv1D has non-finite values (CUDA rc {rc})").into());
    }
    Ok(())
}

fn im2col(
    engine: &Engine,
    input: &CudaSlice<f32>,
    spec: ConvSpec,
    rows: usize,
) -> Result<CudaSlice<f32>, Fail> {
    let mut columns = engine.uninit(rows * spec.in_channels * 3)?;
    let stream = engine.stream();
    let (input_ptr, input_guard) = input.device_ptr(&stream);
    let (columns_ptr, columns_guard) = columns.device_ptr_mut(&stream);
    let rc = unsafe {
        memra_mimo_codec_im2col_bf16(
            input_ptr as *const f32,
            columns_ptr as *mut f32,
            spec.frames as i32,
            spec.in_channels as i32,
            spec.stride as i32,
            rows as i32,
            stream.cu_stream() as *mut c_void,
        )
    };
    drop((input_guard, columns_guard));
    if rc != 0 {
        return Err(format!("MiMo codec BF16 im2col returned {rc}").into());
    }
    Ok(columns)
}

fn epilogue(
    engine: &Engine,
    projected: &CudaSlice<f32>,
    bias: &CudaSlice<u8>,
    rows: usize,
    out_channels: usize,
) -> Result<CudaSlice<f32>, Fail> {
    let mut output = engine.uninit(rows * out_channels)?;
    let stream = engine.stream();
    let (projected_ptr, projected_guard) = projected.device_ptr(&stream);
    let (bias_ptr, bias_guard) = bias.device_ptr(&stream);
    let (output_ptr, output_guard) = output.device_ptr_mut(&stream);
    let rc = unsafe {
        memra_mimo_codec_conv_epilogue(
            projected_ptr as *const f32,
            bias_ptr as *const u16,
            output_ptr as *mut f32,
            rows as i32,
            out_channels as i32,
            stream.cu_stream() as *mut c_void,
        )
    };
    drop((projected_guard, bias_guard, output_guard));
    if rc != 0 {
        return Err(format!("MiMo codec Conv1D GELU epilogue returned {rc}").into());
    }
    Ok(output)
}

fn direct(
    engine: &Engine,
    input: &CudaSlice<f32>,
    weight: &CudaSlice<u8>,
    bias: &CudaSlice<u8>,
    spec: ConvSpec,
    rows: usize,
) -> Result<CudaSlice<f32>, Fail> {
    let mut output = engine.uninit(rows * spec.out_channels)?;
    let stream = engine.stream();
    let (input_ptr, input_guard) = input.device_ptr(&stream);
    let (weight_ptr, weight_guard) = weight.device_ptr(&stream);
    let (bias_ptr, bias_guard) = bias.device_ptr(&stream);
    let (output_ptr, output_guard) = output.device_ptr_mut(&stream);
    let rc = unsafe {
        memra_mimo_codec_conv_direct(
            input_ptr as *const f32,
            weight_ptr as *const u16,
            bias_ptr as *const u16,
            output_ptr as *mut f32,
            spec.frames as i32,
            spec.in_channels as i32,
            spec.out_channels as i32,
            spec.stride as i32,
            rows as i32,
            stream.cu_stream() as *mut c_void,
        )
    };
    drop((input_guard, weight_guard, bias_guard, output_guard));
    if rc != 0 {
        return Err(format!("MiMo codec direct BF16 Conv1D returned {rc}").into());
    }
    Ok(output)
}

fn run_conv(
    engine: &Engine,
    input: &CudaSlice<f32>,
    weight: &CudaSlice<u8>,
    bias: &CudaSlice<u8>,
    spec: ConvSpec,
    force_direct: bool,
) -> Result<CudaSlice<f32>, Fail> {
    let rows = spec.validate(input.len(), weight.len(), bias.len())?;
    engine.gpu.ctx.bind_to_thread()?;
    let ordinal = engine.stream().context().ordinal();
    if input.ordinal() != ordinal || weight.ordinal() != ordinal || bias.ordinal() != ordinal {
        return Err("MiMo codec Conv1D crossed GPU devices".into());
    }
    ensure_finite(engine, input)?;
    // The safetensors weight row is `[out,in,3]`, exactly the contiguous
    // `[out,in*3]` matrix consumed by the existing BF16 GEMM.
    let output = if force_direct {
        direct(engine, input, weight, bias, spec, rows)?
    } else {
        let columns = im2col(engine, input, spec, rows)?;
        match engine.bf16_tc_gemm(
            weight,
            &columns,
            rows,
            spec.in_channels * 3,
            spec.out_channels,
        )? {
            Some(projected) => epilogue(engine, &projected, bias, rows, spec.out_channels)?,
            None => direct(engine, input, weight, bias, spec, rows)?,
        }
    };
    ensure_finite(engine, &output)?;
    Ok(output)
}

impl MiMoAudioCodecEncoderWeights {
    /// From one frame-major `[mel_frames,128]` f32 mel plane, execute the two
    /// pinned BF16 convolutions and erf GELUs. The input is cast to BF16 before
    /// conv1. The result is `[ceil(mel_frames/2),1024]` frame-major f32
    /// carrying BF16 values. At most 1,024 mel frames are admitted per call.
    pub fn encode_prepared_mel_conv(
        &self,
        engine: &Engine,
        mel: &CudaSlice<f32>,
        mel_frames: usize,
    ) -> Result<CudaSlice<f32>, Fail> {
        self.encode_prepared_mel_conv_mode(engine, mel, mel_frames, false)
    }

    #[cfg(test)]
    pub(crate) fn encode_prepared_mel_conv_direct(
        &self,
        engine: &Engine,
        mel: &CudaSlice<f32>,
        mel_frames: usize,
    ) -> Result<CudaSlice<f32>, Fail> {
        self.encode_prepared_mel_conv_mode(engine, mel, mel_frames, true)
    }

    fn encode_prepared_mel_conv_mode(
        &self,
        engine: &Engine,
        mel: &CudaSlice<f32>,
        mel_frames: usize,
        force_direct: bool,
    ) -> Result<CudaSlice<f32>, Fail> {
        if self.device_ordinal != engine.stream().context().ordinal() {
            return Err("MiMo codec encoder weight GPU changed".into());
        }
        let first = self.conv1_bf16()?;
        let second = self.conv2_bf16()?;
        let hidden = run_conv(
            engine,
            mel,
            first.weight(),
            first.bias(),
            ConvSpec {
                frames: mel_frames,
                in_channels: MEL_CHANNELS,
                out_channels: HIDDEN,
                stride: first.stride(),
            },
            force_direct,
        )?;
        run_conv(
            engine,
            &hidden,
            second.weight(),
            second.bias(),
            ConvSpec {
                frames: mel_frames,
                in_channels: HIDDEN,
                out_channels: HIDDEN,
                stride: second.stride(),
            },
            force_direct,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memra_reference::mimo_audio_codec_frontend::conv1d_gelu_bf16;

    #[test]
    fn pinned_conv_geometry_and_bounds() {
        let first = ConvSpec {
            frames: 5,
            in_channels: MEL_CHANNELS,
            out_channels: HIDDEN,
            stride: 1,
        };
        assert_eq!(
            first.validate(5 * MEL_CHANNELS, HIDDEN * MEL_CHANNELS * 6, HIDDEN * 2),
            Ok(5)
        );
        let second = ConvSpec {
            frames: 5,
            in_channels: HIDDEN,
            out_channels: HIDDEN,
            stride: 2,
        };
        assert_eq!(
            second.validate(5 * HIDDEN, HIDDEN * HIDDEN * 6, HIDDEN * 2),
            Ok(3)
        );
        assert!(
            first
                .validate(5 * MEL_CHANNELS - 1, HIDDEN * MEL_CHANNELS * 6, HIDDEN * 2)
                .is_err()
        );
        assert!(
            second
                .validate(5 * HIDDEN, HIDDEN * HIDDEN * 6 - 2, HIDDEN * 2)
                .is_err()
        );
        let excessive = ConvSpec {
            frames: 1_025,
            ..first
        };
        assert!(
            excessive
                .validate(1_025 * MEL_CHANNELS, HIDDEN * MEL_CHANNELS * 6, HIDDEN * 2)
                .is_err()
        );
    }

    #[test]
    #[ignore = "requires a dedicated MiMo GPU component lane"]
    fn gpu_two_conv_frontend_matches_portable_oracle() -> Result<(), Fail> {
        let gpu: usize = std::env::var("MEMRA_MIMO_COMPONENT_GPU")
            .unwrap_or_else(|_| "0".into())
            .parse()?;
        let engine = Engine::new(gpu)?;
        let input = (0..10)
            .map(|index| (index as f32 - 4.0) * 0.125)
            .collect::<Vec<_>>();
        let to_bf16 = |value: f32| (value.to_bits() >> 16) as u16;
        let weight1 = (0..18)
            .map(|index| to_bf16((index as f32 - 8.0) * 0.0625))
            .collect::<Vec<_>>();
        let weight2 = (0..18)
            .map(|index| to_bf16((index as f32 - 9.0) * 0.03125))
            .collect::<Vec<_>>();
        let bias1 = [to_bf16(0.125), to_bf16(-0.0625), to_bf16(0.25)];
        let bias2 = [to_bf16(0.03125), to_bf16(-0.125)];
        let bytes = |values: &[u16]| {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
        };
        let expected1 = conv1d_gelu_bf16(&input, 5, 2, 3, 1, &weight1, &bias1)?;
        let expected2 = conv1d_gelu_bf16(&expected1, 5, 3, 2, 2, &weight2, &bias2)?;
        let actual1 = run_conv(
            &engine,
            &engine.htod(&input)?,
            &engine.htod_bytes(&bytes(&weight1))?,
            &engine.htod_bytes(&bytes(&bias1))?,
            ConvSpec {
                frames: 5,
                in_channels: 2,
                out_channels: 3,
                stride: 1,
            },
            false,
        )?;
        let actual2 = run_conv(
            &engine,
            &actual1,
            &engine.htod_bytes(&bytes(&weight2))?,
            &engine.htod_bytes(&bytes(&bias2))?,
            ConvSpec {
                frames: 5,
                in_channels: 3,
                out_channels: 2,
                stride: 2,
            },
            false,
        )?;
        let actual = engine.dtoh(&actual2)?;
        assert_eq!(actual.len(), expected2.len());
        for (index, (&got, &want)) in actual.iter().zip(&expected2).enumerate() {
            assert!(
                (got - want).abs() <= 0.015625,
                "element {index}: {got} != {want}"
            );
        }
        Ok(())
    }
}
