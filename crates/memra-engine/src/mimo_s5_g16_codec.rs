//! Standalone signed uniform 5-bit MiMo V component codec.
//!
//! Each [128] head row contains eight 16-value groups. A group is one UE4M3
//! scale byte followed by 80 little-endian packed code bits: 88 bytes per
//! head row, 352 bytes per [4,128] token row. Codes are two's complement
//! [-16,15]. This module has no KV cache, forward, or serving dispatch.
//!
//! Scale and tie rules match the source-only `value_codec_quality_probe.rs`
//! at ece94bf4a113b7d5dd6d73a808e8279580fa00d5.

use core::ffi::c_void;
use std::error::Error;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use memra_gguf::nvfp4_repack::ue4m3_to_f32;

use crate::Engine;

pub const HEAD_WIDTH: usize = 128;
pub const GROUP_WIDTH: usize = 16;
pub const GROUP_BYTES: usize = 11;
pub const HEAD_BYTES: usize = HEAD_WIDTH / GROUP_WIDTH * GROUP_BYTES;
pub const TOKEN_BYTES: usize = 4 * HEAD_BYTES;
/// Covers at most 256 four-head token rows per component call.
pub const MAX_COMPONENT_ROWS: usize = 1024;

unsafe extern "C" {
    fn memra_mimo_s5_g16_encode_f32(
        input: *const f32,
        input_elements: usize,
        output: *mut u8,
        output_bytes: usize,
        rows: usize,
        stream: *mut c_void,
    ) -> i32;
    fn memra_mimo_s5_g16_decode_f32(
        input: *const u8,
        input_bytes: usize,
        output: *mut f32,
        output_elements: usize,
        rows: usize,
        stream: *mut c_void,
    ) -> i32;
}

pub fn packed_bytes(rows: usize) -> Result<usize, &'static str> {
    if !(1..=MAX_COMPONENT_ROWS).contains(&rows) {
        return Err("MiMo S5 component requires 1..1024 head rows");
    }
    rows.checked_mul(HEAD_BYTES)
        .ok_or("MiMo S5 packed extent overflowed")
}

fn input_rows(elements: usize) -> Result<usize, &'static str> {
    if !elements.is_multiple_of(HEAD_WIDTH) {
        return Err("MiMo S5 input must contain complete [128] head rows");
    }
    let rows = elements / HEAD_WIDTH;
    packed_bytes(rows)?;
    Ok(rows)
}

// Probe rule: search every positive finite UE4M3 code in ascending order;
// strict comparison leaves the lower code on an exact distance tie. The GGUF
// decoder halves scales, so multiply its return value by two.
fn scale_byte(target: f32) -> u8 {
    if !target.is_finite() || target <= 0.0 {
        return 0;
    }
    let mut best = 0u8;
    let mut distance = f32::INFINITY;
    for code in 1u8..0x7f {
        let error = (ue4m3_to_f32(code) * 2.0 - target).abs();
        if error < distance {
            best = code;
            distance = error;
        }
    }
    best
}

/// Portable byte reference for complete [128] f32 head rows.
pub fn encode_reference(input: &[f32]) -> Result<Vec<u8>, &'static str> {
    let rows = input_rows(input.len())?;
    let mut output = Vec::with_capacity(packed_bytes(rows)?);
    for group in input.chunks_exact(GROUP_WIDTH) {
        let positive = group.iter().fold(0.0f32, |max, &x| max.max(x));
        let negative = group.iter().fold(0.0f32, |max, &x| max.max(-x));
        let scale_code = scale_byte((positive / 15.0).max(negative / 16.0));
        let scale = ue4m3_to_f32(scale_code) * 2.0;
        let mut block = [0u8; GROUP_BYTES];
        block[0] = scale_code;
        for (index, &value) in group.iter().enumerate() {
            let code = if scale > 0.0 {
                (value / scale).round().clamp(-16.0, 15.0) as i8
            } else {
                0
            };
            let bits = (code as u8) & 31;
            let bit = index * 5;
            let byte = 1 + bit / 8;
            let shift = bit % 8;
            block[byte] |= bits << shift;
            if shift > 3 {
                block[byte + 1] |= bits >> (8 - shift);
            }
        }
        output.extend_from_slice(&block);
    }
    Ok(output)
}

/// Portable decoder for complete [128] head rows.
pub fn decode_reference(input: &[u8]) -> Result<Vec<f32>, &'static str> {
    if !input.len().is_multiple_of(HEAD_BYTES) {
        return Err("MiMo S5 packed input must contain complete [128] head rows");
    }
    let rows = input.len() / HEAD_BYTES;
    packed_bytes(rows)?;
    let mut output = Vec::with_capacity(rows * HEAD_WIDTH);
    for group in input.chunks_exact(GROUP_BYTES) {
        let scale = ue4m3_to_f32(group[0]) * 2.0;
        for index in 0..GROUP_WIDTH {
            let bit = index * 5;
            let byte = 1 + bit / 8;
            let shift = bit % 8;
            let mut packed = u16::from(group[byte]);
            if shift > 3 {
                packed |= u16::from(group[byte + 1]) << 8;
            }
            let bits = i32::from((packed >> shift) & 31);
            let code = if bits >= 16 { bits - 32 } else { bits };
            output.push(code as f32 * scale);
        }
    }
    Ok(output)
}

impl Engine {
    /// Encode up to 1024 already MiMo-scaled [128] V head rows. The native
    /// launch is checked and completed on this engine's stream.
    pub fn mimo_s5_g16_encode_rows(
        &self,
        input: &CudaSlice<f32>,
    ) -> Result<CudaSlice<u8>, Box<dyn Error>> {
        let rows = input_rows(input.len())?;
        let output_bytes = packed_bytes(rows)?;
        self.gpu.ctx.bind_to_thread()?;
        let stream = self.stream();
        if input.ordinal() != stream.context().ordinal() {
            return Err("MiMo S5 encode input crossed GPU devices".into());
        }
        let mut output = self.alloc_u8_uninit(output_bytes)?;
        let rc = unsafe {
            memra_mimo_s5_g16_encode_f32(
                input.device_ptr(&stream).0 as *const f32,
                input.len(),
                output.device_ptr_mut(&stream).0 as *mut u8,
                output_bytes,
                rows,
                stream.cu_stream() as *mut c_void,
            )
        };
        if rc != 0 {
            return Err(format!("MiMo S5 GPU encode returned {rc}").into());
        }
        stream.synchronize()?;
        Ok(output)
    }

    /// Decode up to 1024 contiguous [128] V head rows.
    pub fn mimo_s5_g16_decode_rows(
        &self,
        input: &CudaSlice<u8>,
        rows: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn Error>> {
        let input_bytes = packed_bytes(rows)?;
        if input.len() != input_bytes {
            return Err("MiMo S5 decode input extent mismatch".into());
        }
        self.gpu.ctx.bind_to_thread()?;
        let stream = self.stream();
        if input.ordinal() != stream.context().ordinal() {
            return Err("MiMo S5 decode input crossed GPU devices".into());
        }
        let output_elements = rows * HEAD_WIDTH;
        let mut output = self.uninit(output_elements)?;
        let rc = unsafe {
            memra_mimo_s5_g16_decode_f32(
                input.device_ptr(&stream).0 as *const u8,
                input_bytes,
                output.device_ptr_mut(&stream).0 as *mut f32,
                output_elements,
                rows,
                stream.cu_stream() as *mut c_void,
            )
        };
        if rc != 0 {
            return Err(format!("MiMo S5 GPU decode returned {rc}").into());
        }
        stream.synchronize()?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn component_extents_and_refusals() {
        assert_eq!(HEAD_BYTES, 88);
        assert_eq!(TOKEN_BYTES, 352);
        assert_eq!(packed_bytes(4).unwrap(), TOKEN_BYTES);
        assert!(packed_bytes(0).is_err());
        assert!(packed_bytes(MAX_COMPONENT_ROWS + 1).is_err());
        assert!(encode_reference(&[]).is_err());
        assert!(encode_reference(&[0.0; HEAD_WIDTH - 1]).is_err());
        assert!(decode_reference(&[]).is_err());
        assert!(decode_reference(&[0; HEAD_BYTES - 1]).is_err());
    }

    #[test]
    fn signed_extremes_have_a_known_byte_pattern() {
        let mut values = [0.0f32; HEAD_WIDTH];
        values[0] = 15.0;
        values[1] = -16.0;
        let encoded = encode_reference(&values).unwrap();
        assert_eq!(
            &encoded[..GROUP_BYTES],
            &[56, 0x0f, 0x02, 0, 0, 0, 0, 0, 0, 0, 0]
        );
        assert!(encoded[GROUP_BYTES..].iter().all(|&byte| byte == 0));
        assert_eq!(decode_reference(&encoded).unwrap(), values);
    }

    #[test]
    fn scale_and_code_ties_are_deterministic() {
        // Raw scale codes 1 and 2 are 1/512 and 2/512.
        assert_eq!(scale_byte(3.0 / 1024.0), 1);
        assert_eq!(scale_byte(f32::NAN), 0);
        assert_eq!(scale_byte(f32::INFINITY), 0);
        let mut values = [0.0f32; HEAD_WIDTH];
        values[0] = 1.0 / 1024.0;
        values[1] = -1.0 / 1024.0;
        let encoded = encode_reference(&values).unwrap();
        assert_eq!(&encoded[..3], &[1, 0xe1, 0x03]);
        let decoded = decode_reference(&encoded).unwrap();
        assert_eq!(decoded[0], 1.0 / 512.0);
        assert_eq!(decoded[1], -1.0 / 512.0);
    }

    #[test]
    fn four_head_rows_are_independent() {
        let mut values = vec![0.0f32; 4 * HEAD_WIDTH];
        for head in 0..4 {
            values[head * HEAD_WIDTH] = 15.0 * (head + 1) as f32;
            values[head * HEAD_WIDTH + 15] = -16.0 * (head + 1) as f32;
        }
        let encoded = encode_reference(&values).unwrap();
        assert_eq!(encoded.len(), TOKEN_BYTES);
        for (head, source) in values.chunks_exact(HEAD_WIDTH).enumerate() {
            let start = head * HEAD_BYTES;
            assert_eq!(
                &encoded[start..start + HEAD_BYTES],
                &encode_reference(source).unwrap()
            );
        }
        let decoded = decode_reference(&encoded).unwrap();
        assert_eq!(decoded.len(), values.len());
        for head in 0..4 {
            assert_eq!(decoded[head * HEAD_WIDTH], values[head * HEAD_WIDTH]);
            assert_eq!(
                decoded[head * HEAD_WIDTH + 15],
                values[head * HEAD_WIDTH + 15]
            );
        }
    }

    #[test]
    fn decode_sign_extends_last_five_bit_code() {
        let mut bytes = [0u8; HEAD_BYTES];
        bytes[0] = 56; // Raw scale = 1.
        bytes[10] = 0x80; // Code 16 at index 15 is -16.
        let decoded = decode_reference(&bytes).unwrap();
        assert!(decoded[..15].iter().all(|&value| value == 0.0));
        assert_eq!(decoded[15], -16.0);
        assert!(decoded[16..].iter().all(|&value| value == 0.0));
    }

    #[test]
    #[ignore = "requires an explicitly assigned GPU; component parity is unqualified here"]
    fn gpu_component_roundtrip_matches_reference() -> Result<(), Box<dyn Error>> {
        let engine = Engine::new(0)?;
        for rows in [1, 4, 512] {
            let values = (0..rows * HEAD_WIDTH)
                .map(|index| {
                    let choices = [
                        0.0,
                        -0.0,
                        1.0 / 1024.0,
                        -1.0 / 1024.0,
                        0.5,
                        -0.5,
                        15.0,
                        -16.0,
                        448.0,
                        -448.0,
                    ];
                    choices[(index * 17 + index / GROUP_WIDTH) % choices.len()]
                })
                .collect::<Vec<_>>();
            let expected_bytes = encode_reference(&values)?;
            let expected_values = decode_reference(&expected_bytes)?;
            let input = engine.htod(&values)?;
            let packed = engine.mimo_s5_g16_encode_rows(&input)?;
            assert_eq!(engine.dtoh_u8(&packed)?, expected_bytes);
            let decoded = engine.mimo_s5_g16_decode_rows(&packed, rows)?;
            let actual = engine.dtoh(&decoded)?;
            assert!(
                actual
                    .iter()
                    .zip(expected_values.iter())
                    .all(|(got, want)| got.to_bits() == want.to_bits())
            );
        }
        Ok(())
    }
}
