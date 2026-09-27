//! Bounded source-order RVQ over post-downsample F32 MiMo codec rows.

use core::ffi::c_void;
use std::error::Error;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use memra_reference::mimo_audio::GroupedAudioCodes;
use memra_reference::mimo_audio_codec_rvq::{
    MAX_TOKENS, RVQ_DEPTHS, expected_bins, group_for_patch,
};

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

/// Complete source RVQ encode. The IDs are host-visible token-major `[T,20]`,
/// with ID `(token,depth)` at `code_ids[token*20+depth]`. Pinned Python returns
/// `[20,T]`, with the same ID at `source[depth*T+token]`.
pub struct Rvq20DeviceResult {
    pub code_ids: Vec<u16>,
    pub residual: CudaSlice<f32>,
    pub tokens: usize,
}

impl Rvq20DeviceResult {
    /// Repeat the final row to make `[groups,4,20]` for patch intake.
    pub fn grouped_patch_codes(&self) -> Result<GroupedAudioCodes, String> {
        group_for_patch(&self.code_ids, self.tokens)
    }
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

    /// Run pinned `ResidualVectorQuantization.encode` at depths `0..19`.
    ///
    /// Input is already post-downsampled contiguous F32 `[tokens,1024]`.
    /// This borrows only model-owned F32 codebooks. The loader checked their
    /// payload finiteness; all twenty pinned shapes, bin counts, and devices
    /// are rechecked before the first GPU operation. No partial result or
    /// mutable encode state is retained if a later depth fails.
    pub fn encode_20_rvq(
        &self,
        engine: &Engine,
        input: &CudaSlice<f32>,
        tokens: usize,
    ) -> Result<Rvq20DeviceResult, Fail> {
        self.check_device(engine)?;
        if !(1..=MAX_TOKENS).contains(&tokens) || input.len() != tokens * WIDTH {
            return Err("MiMo RVQ twenty-depth token count or F32 extent changed".into());
        }
        let ordinal = engine.stream().context().ordinal();
        if input.ordinal() != ordinal {
            return Err("MiMo RVQ twenty-depth input crossed GPU devices".into());
        }
        let mut codebooks = Vec::with_capacity(RVQ_DEPTHS);
        for depth in 0..RVQ_DEPTHS {
            let codebook = self.codebook_f32(depth)?;
            if expected_bins(depth) != Some(codebook.bins) {
                return Err(format!("MiMo RVQ depth {depth} bin schedule changed").into());
            }
            request(tokens, input.len(), codebook.bytes.len(), codebook.bins)?;
            codebooks.push(codebook);
        }

        let outcome = (|| {
            let mut residual: Option<CudaSlice<f32>> = None;
            let mut code_ids = vec![0u16; tokens * RVQ_DEPTHS];
            for (depth, codebook) in codebooks.iter().enumerate() {
                let current = residual.as_ref().unwrap_or(input);
                let step = run_codebook(engine, current, tokens, codebook.bytes, codebook.bins)?;
                let ids = engine.dtoh_i32(&step.code_ids)?;
                if ids.len() != tokens {
                    return Err(format!("MiMo RVQ depth {depth} returned wrong ID extent").into());
                }
                for (token, id) in ids.into_iter().enumerate() {
                    if id < 0 || id as usize >= codebook.bins {
                        return Err(format!("MiMo RVQ depth {depth} returned invalid ID").into());
                    }
                    code_ids[token * RVQ_DEPTHS + depth] = id as u16;
                }
                residual = Some(step.residual);
            }
            Ok(Rvq20DeviceResult {
                code_ids,
                residual: residual.expect("twenty checked RVQ depths"),
                tokens,
            })
        })();
        if outcome.is_err() {
            // The operation is stateless and no incomplete IDs escape.
            // Device allocations free on their stream; drain it before
            // another call can use this engine after a partial failure.
            let _ = engine.stream().synchronize();
        }
        outcome
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PINNED_SOURCE_PYTHON: &str = r#"
import ast
import struct
import sys
from pathlib import Path
from typing import Optional

import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F

source_path, root, gpu = sys.argv[1:]
source = ast.parse(Path(source_path).read_text())
classes = {"EuclideanCodebook", "VectorQuantization", "ResidualVectorQuantization"}
definitions = [
    node for node in source.body
    if isinstance(node, ast.ClassDef) and node.name in classes
]
if len(definitions) != 3:
    raise RuntimeError("pinned RVQ source classes changed")
scope = {
    "__name__": "mimo_pinned_rvq",
    "torch": torch, "nn": nn, "F": F, "Optional": Optional,
}
exec(compile(ast.Module(body=definitions, type_ignores=[]), source_path, "exec"), scope)

device = torch.device(f"cuda:{gpu}")
bins = [1024, 1024, 256] + [128] * 17
rvq = scope["ResidualVectorQuantization"](
    num_quantizers=20, codebook_size=bins, dim=1024, kmeans_init=True
).to(device)
with safe_open(str(Path(root) / "audio_tokenizer/model.safetensors"), framework="pt", device="cpu") as weights:
    with torch.no_grad():
        for depth, layer in enumerate(rvq.layers):
            name = f"encoder.quantizer.vq.layers.{depth}._codebook.embed"
            embed = weights.get_tensor(name)
            if embed.dtype != torch.float32 or tuple(embed.shape) != (bins[depth], 1024):
                raise RuntimeError(f"{name} F32 shape changed")
            layer._codebook.embed.copy_(embed.to(device))

values = [((index % 47) - 23) * 0.015625 for index in range(2 * 1024)]
input_rows = torch.tensor(values, dtype=torch.float32, device=device).reshape(2, 1024)
with torch.no_grad():
    codes = rvq.encode(input_rows)
    residual = input_rows
    for depth, layer in enumerate(rvq.layers):
        residual = residual - layer.decode(codes[depth])
print(" ".join(str(int(code)) for code in codes.cpu().flatten().tolist()))
print(" ".join(
    str(struct.unpack("<I", struct.pack("<f", value))[0])
    for value in residual.cpu().flatten().tolist()
))
"#;

    fn pinned_source_gpu_result(
        root: &std::path::Path,
        gpu: usize,
    ) -> Result<(Vec<u16>, Vec<u32>), Fail> {
        use sha2::{Digest, Sha256};
        use std::process::Command;

        let source_path = std::env::var("MIMO_PINNED_MODELING_PY")
            .unwrap_or_else(|_| "/tmp/mimo-pinned-source-20260926/modeling_mimo_v2.py".into());
        let hash = format!("{:x}", Sha256::digest(std::fs::read(&source_path)?));
        if hash != "a8c3cb3aae473bcc15f023010547c919f15eba6546e6ed7efb61a8937b12f3ad" {
            return Err("MiMo RVQ Python source hash changed".into());
        }
        let python = std::env::var("MIMO_PINNED_PYTHON").unwrap_or_else(|_| "python3".into());
        let output = Command::new(python)
            .arg("-c")
            .arg(PINNED_SOURCE_PYTHON)
            .arg(source_path)
            .arg(root)
            .arg(gpu.to_string())
            .output()?;
        if !output.status.success() {
            return Err(format!(
                "MiMo RVQ pinned Python source failed on GPU {gpu}: {}",
                String::from_utf8_lossy(&output.stderr)
            )
            .into());
        }
        let stdout = String::from_utf8(output.stdout)?;
        let mut lines = stdout.lines();
        let ids = lines
            .next()
            .ok_or("MiMo RVQ pinned Python source omitted IDs")?
            .split_whitespace()
            .map(str::parse::<u16>)
            .collect::<Result<Vec<_>, _>>()?;
        let residual = lines
            .next()
            .ok_or("MiMo RVQ pinned Python source omitted residual")?
            .split_whitespace()
            .map(str::parse::<u32>)
            .collect::<Result<Vec<_>, _>>()?;
        if lines.next().is_some() {
            return Err("MiMo RVQ pinned Python source printed extra output".into());
        }
        Ok((ids, residual))
    }

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
    #[ignore = "requires pinned source Python, bundled weights, and two delegated target GPUs"]
    fn pinned_two_card_twenty_depth_gpu_source_gate() -> Result<(), Fail> {
        use std::path::Path;

        let root = std::env::var("MIMO_PINNED_SOURCE_ROOT")?;
        let root = Path::new(&root);
        let gpu_ids = std::env::var("MEMRA_MIMO_COMPONENT_GPUS")?
            .split(',')
            .map(str::parse::<usize>)
            .collect::<Result<Vec<_>, _>>()?;
        if gpu_ids.len() != 2 || gpu_ids[0] == gpu_ids[1] {
            return Err("MiMo RVQ source gate needs two distinct GPU ordinals".into());
        }
        let tokens = 2;
        let input = (0..tokens * WIDTH)
            .map(|index| (index as f32 % 47.0 - 23.0) * 0.015625)
            .collect::<Vec<_>>();
        for gpu in gpu_ids {
            let engine = Engine::new(gpu)?;
            let resident = MiMoAudioCodecEncoderWeights::load(&engine, root)?;
            let got = resident.encode_20_rvq(&engine, &engine.htod(&input)?, tokens)?;
            let next = engine.dtoh(&got.residual)?;
            let (source_ids, source_residual_bits) = pinned_source_gpu_result(root, gpu)?;
            assert_eq!(source_ids.len(), RVQ_DEPTHS * tokens);
            assert_eq!(source_residual_bits.len(), tokens * WIDTH);
            for depth in 0..RVQ_DEPTHS {
                for token in 0..tokens {
                    assert_eq!(
                        got.code_ids[token * RVQ_DEPTHS + depth],
                        source_ids[depth * tokens + token],
                        "GPU {gpu}, depth {depth}, token {token}"
                    );
                }
            }
            for (index, &actual) in next.iter().enumerate() {
                assert_eq!(
                    actual.to_bits(),
                    source_residual_bits[index],
                    "GPU {gpu}, source residual element {index}"
                );
            }
            let grouped = got.grouped_patch_codes()?;
            assert_eq!(grouped.groups, 1);
            assert_eq!(&grouped.codes[..40], &got.code_ids);
            assert_eq!(&grouped.codes[40..60], &got.code_ids[20..40]);
            assert_eq!(&grouped.codes[60..80], &got.code_ids[20..40]);
        }
        Ok(())
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
