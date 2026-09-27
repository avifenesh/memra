//! Bounded one-codebook RVQ step over post-downsample F32 MiMo codec rows.
//!
//! This runs one selected depth. It does not iterate the 20 RVQ layers.

use core::ffi::c_void;
use std::error::Error;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use memra_reference::mimo_audio_codec_rvq::MAX_TOKENS;

use crate::Engine;
use crate::mimo_audio_codec_weights::MiMoAudioCodecEncoderWeights;

type Fail = Box<dyn Error>;
const WIDTH: usize = 1_024;

unsafe extern "C" {
    fn memra_mimo_codec_rvq_check_finite(
        values: *const f32,
        count: i32,
        fault: *mut i32,
        stream: *mut c_void,
    ) -> i32;
    fn memra_mimo_codec_rvq_select(
        residual: *const f32,
        embed: *const f32,
        code_ids: *mut i32,
        tokens: i32,
        bins: i32,
        fault: *mut i32,
        stream: *mut c_void,
    ) -> i32;
    fn memra_mimo_codec_rvq_subtract(
        residual: *const f32,
        embed: *const f32,
        code_ids: *const i32,
        next: *mut f32,
        tokens: i32,
        bins: i32,
        fault: *mut i32,
        stream: *mut c_void,
    ) -> i32;
}

/// Device outputs for one source RVQ layer.
pub struct OneCodebookDeviceResult {
    pub code_ids: CudaSlice<i32>,
    pub residual: CudaSlice<f32>,
}

fn request(
    tokens: usize,
    residual_len: usize,
    embed_bytes: usize,
    bins: usize,
) -> Result<(), Fail> {
    if !(1..=MAX_TOKENS).contains(&tokens)
        || residual_len != tokens * WIDTH
        || !matches!(bins, 128 | 256 | 1_024)
        || embed_bytes != bins * WIDTH * 4
    {
        return Err("MiMo RVQ one-codebook token count or F32 extent changed".into());
    }
    Ok(())
}

fn checked_rc(operation: &str, rc: i32) -> Result<(), Fail> {
    if rc != 0 {
        return Err(format!("MiMo RVQ {operation} CUDA refusal {rc}").into());
    }
    Ok(())
}

fn check_finite(
    engine: &Engine,
    values: &CudaSlice<f32>,
    fault: &mut CudaSlice<i32>,
    operation: &str,
) -> Result<(), Fail> {
    let stream = engine.stream();
    let (values_ptr, values_guard) = values.device_ptr(&stream);
    let (fault_ptr, fault_guard) = fault.device_ptr_mut(&stream);
    let rc = unsafe {
        memra_mimo_codec_rvq_check_finite(
            values_ptr as *const f32,
            values.len() as i32,
            fault_ptr as *mut i32,
            stream.cu_stream() as *mut c_void,
        )
    };
    drop((values_guard, fault_guard));
    checked_rc(operation, rc)
}

fn run_codebook(
    engine: &Engine,
    residual: &CudaSlice<f32>,
    tokens: usize,
    embed: &CudaSlice<u8>,
    bins: usize,
) -> Result<OneCodebookDeviceResult, Fail> {
    request(tokens, residual.len(), embed.len(), bins)?;
    engine.gpu.ctx.bind_to_thread()?;
    let stream = engine.stream();
    let ordinal = stream.context().ordinal();
    if residual.ordinal() != ordinal || embed.ordinal() != ordinal {
        return Err("MiMo RVQ one-codebook crossed GPU devices".into());
    }

    let mut fault = engine.htod_i32(&[0])?;
    check_finite(engine, residual, &mut fault, "input finite check")?;
    if engine.dtoh_i32(&fault)? != [0] {
        return Err("MiMo RVQ input has non-finite F32 values".into());
    }

    let mut code_ids = engine.uninit_i32(tokens)?;
    let (residual_ptr, residual_guard) = residual.device_ptr(&stream);
    let (embed_ptr, embed_guard) = embed.device_ptr(&stream);
    let (ids_ptr, ids_guard) = code_ids.device_ptr_mut(&stream);
    let (fault_ptr, fault_guard) = fault.device_ptr_mut(&stream);
    let rc = unsafe {
        memra_mimo_codec_rvq_select(
            residual_ptr as *const f32,
            embed_ptr as *const f32,
            ids_ptr as *mut i32,
            tokens as i32,
            bins as i32,
            fault_ptr as *mut i32,
            stream.cu_stream() as *mut c_void,
        )
    };
    drop((residual_guard, embed_guard, ids_guard, fault_guard));
    checked_rc("code selection", rc)?;
    if engine.dtoh_i32(&fault)? != [0] {
        return Err("MiMo RVQ source distance became non-finite".into());
    }

    let mut next = engine.uninit(residual.len())?;
    let (residual_ptr, residual_guard) = residual.device_ptr(&stream);
    let (embed_ptr, embed_guard) = embed.device_ptr(&stream);
    let (ids_ptr, ids_guard) = code_ids.device_ptr(&stream);
    let (next_ptr, next_guard) = next.device_ptr_mut(&stream);
    let (fault_ptr, fault_guard) = fault.device_ptr_mut(&stream);
    let rc = unsafe {
        memra_mimo_codec_rvq_subtract(
            residual_ptr as *const f32,
            embed_ptr as *const f32,
            ids_ptr as *const i32,
            next_ptr as *mut f32,
            tokens as i32,
            bins as i32,
            fault_ptr as *mut i32,
            stream.cu_stream() as *mut c_void,
        )
    };
    drop((
        residual_guard,
        embed_guard,
        ids_guard,
        next_guard,
        fault_guard,
    ));
    checked_rc("residual subtraction", rc)?;
    check_finite(engine, &next, &mut fault, "residual finite check")?;
    if engine.dtoh_i32(&fault)? != [0] {
        return Err("MiMo RVQ decoded index or next residual is invalid".into());
    }
    Ok(OneCodebookDeviceResult {
        code_ids,
        residual: next,
    })
}

impl MiMoAudioCodecEncoderWeights {
    /// Source `ResidualVectorQuantization.encode` for one selected depth only.
    /// `residual` is already post-downsampled contiguous F32 `[tokens,1024]`.
    pub fn encode_one_rvq_step(
        &self,
        engine: &Engine,
        residual: &CudaSlice<f32>,
        tokens: usize,
        depth: usize,
    ) -> Result<OneCodebookDeviceResult, Fail> {
        self.check_device(engine)?;
        let codebook = self.codebook_f32(depth)?;
        run_codebook(engine, residual, tokens, codebook.bytes, codebook.bins)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_component_refuses_bad_extents() {
        assert!(request(0, 0, 128 * WIDTH * 4, 128).is_err());
        assert!(
            request(
                MAX_TOKENS + 1,
                (MAX_TOKENS + 1) * WIDTH,
                128 * WIDTH * 4,
                128
            )
            .is_err()
        );
        assert!(request(1, WIDTH - 1, 128 * WIDTH * 4, 128).is_err());
        assert!(request(1, WIDTH, 128 * WIDTH * 4 - 4, 128).is_err());
        assert!(request(1, WIDTH, 129 * WIDTH * 4, 129).is_err());
        assert!(request(1, WIDTH, 128 * WIDTH * 4, 128).is_ok());
    }

    #[test]
    #[ignore = "requires pinned bundled weights and a dedicated target GPU"]
    fn pinned_one_codebook_gpu_source_gate() -> Result<(), Fail> {
        use memra_reference::mimo_audio_codec_rvq::encode_one;
        use std::path::Path;

        let root = std::env::var("MIMO_PINNED_SOURCE_ROOT")?;
        let gpu: usize = std::env::var("MEMRA_MIMO_COMPONENT_GPU")
            .unwrap_or_else(|_| "0".into())
            .parse()?;
        let engine = Engine::new(gpu)?;
        let resident = MiMoAudioCodecEncoderWeights::load(&engine, Path::new(&root))?;
        let tokens = 2;
        let input = (0..tokens * WIDTH)
            .map(|index| (index as f32 % 47.0 - 23.0) * 0.015625)
            .collect::<Vec<_>>();
        for depth in 0..20 {
            let codebook = resident.codebook_f32(depth)?;
            let bytes = engine.dtoh_u8(codebook.bytes)?;
            let embed = bytes
                .chunks_exact(4)
                .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
                .collect::<Vec<_>>();
            let expected = encode_one(&input, tokens, WIDTH, &embed, codebook.bins)?;
            let got =
                resident.encode_one_rvq_step(&engine, &engine.htod(&input)?, tokens, depth)?;
            let ids = engine.dtoh_i32(&got.code_ids)?;
            let next = engine.dtoh(&got.residual)?;
            assert_eq!(
                ids,
                expected
                    .code_ids
                    .iter()
                    .map(|&index| index as i32)
                    .collect::<Vec<_>>(),
                "depth {depth}"
            );
            for (index, (&actual, &want)) in next.iter().zip(&expected.residual).enumerate() {
                assert_eq!(
                    actual.to_bits(),
                    want.to_bits(),
                    "depth {depth}, residual element {index}"
                );
            }
        }
        Ok(())
    }
}
