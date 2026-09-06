//! CPU reference for the experimental row-scaled latent NVFP4 codec.
//! This is not a weight-mint recipe and does not enable a serving path.
pub use crate::latent_layout::LatentBasis;
use crate::latent_layout::{LatentFormat, LatentLayout};
use memra_gguf::nvfp4_repack::{f32_to_fp8_e4m3, fp8_e4m3_to_f32};
use std::sync::{Arc, Mutex, MutexGuard};

const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// One sticky status word per request/stage, shared by that stage's latent planes.
#[derive(Clone)]
pub struct DeviceStatus(Arc<Mutex<StatusState>>);
pub struct StatusState {
    pub buffer: cudarc::driver::CudaSlice<i32>,
    dirty: bool,
    code: i32,
}
impl DeviceStatus {
    pub fn new(e: &dyn crate::KvDev) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self(Arc::new(Mutex::new(StatusState {
            buffer: e.htod_i32(&[0])?,
            dirty: false,
            code: 0,
        }))))
    }
    pub fn identity(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }
    pub fn write_buffer(&self) -> Result<MutexGuard<'_, StatusState>, Box<dyn std::error::Error>> {
        let mut state = self.0.lock().map_err(|_| "poisoned latent status lock")?;
        if state.buffer.len() != 1 {
            return Err("invalid latent status extent".into());
        }
        state.dirty = true;
        Ok(state)
    }
    /// Captured graph launches do not pass through the normal kernel wrappers.
    pub fn invalidate(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.0
            .lock()
            .map_err(|_| "poisoned latent status lock")?
            .dirty = true;
        Ok(())
    }
    pub fn check(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut state = self.0.lock().map_err(|_| "poisoned latent status lock")?;
        if state.buffer.len() != 1 {
            return Err("invalid latent status extent".into());
        }
        if state.dirty {
            state.code = state.buffer.stream().clone_dtoh(&state.buffer)?[0];
            state.dirty = false;
        }
        if state.code != 0 {
            return Err(format!("NVFP4 latent device validation failed: {}", state.code).into());
        }
        Ok(())
    }
}

/// Resident append-only cache representation. Device kernels live in memra-engine.
/// Layout is fixed for the lifetime of a plane and every prefix snapshot.
pub struct DevicePlane {
    pub payload: cudarc::driver::CudaSlice<u8>,
    pub scales: cudarc::driver::CudaSlice<u8>,
    pub macros: cudarc::driver::CudaSlice<f32>,
    pub error: DeviceStatus,
    width: usize,
    capacity: usize,
    basis: LatentBasis,
}

impl DevicePlane {
    pub fn new(
        e: &dyn crate::KvDev,
        width: usize,
        capacity: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_with_basis(e, width, capacity, LatentBasis::Identity)
    }
    pub fn new_with_basis(
        e: &dyn crate::KvDev,
        width: usize,
        capacity: usize,
        basis: LatentBasis,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        basis.validate(width)?;
        Self::with_status_basis(e, width, capacity, DeviceStatus::new(e)?, basis)
    }
    pub fn with_status(
        e: &dyn crate::KvDev,
        width: usize,
        capacity: usize,
        error: DeviceStatus,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::with_status_basis(e, width, capacity, error, LatentBasis::Identity)
    }
    pub fn with_status_basis(
        e: &dyn crate::KvDev,
        width: usize,
        capacity: usize,
        error: DeviceStatus,
        basis: LatentBasis,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let layout = LatentLayout::new_with_basis(LatentFormat::Nvfp4, width, basis)?;
        i32::try_from(width)?;
        i32::try_from(capacity)?;
        layout.allocation_bytes(capacity)?;
        Ok(Self {
            payload: e.alloc_u8(layout.payload_bytes * capacity)?,
            scales: e.alloc_u8(layout.block_scale_bytes * capacity)?,
            // Zero macro scales make uninitialized history an explicit invalid read.
            macros: e.zeros(capacity)?,
            error,
            width,
            capacity,
            basis,
        })
    }

    pub fn width(&self) -> usize {
        self.width
    }
    pub fn basis(&self) -> LatentBasis {
        self.basis
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn allocated_bytes(&self) -> usize {
        self.payload.len() + self.scales.len() + self.macros.len() * 4 + 4
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        let layout = LatentLayout::new_with_basis(LatentFormat::Nvfp4, self.width, self.basis)?;
        layout.allocation_bytes(self.capacity)?;
        if self.payload.len() != layout.payload_bytes * self.capacity
            || self.scales.len() != layout.block_scale_bytes * self.capacity
            || self.macros.len() != self.capacity
        {
            return Err("truncated or inconsistent NVFP4 latent planes");
        }
        Ok(())
    }

    pub fn copy_prefix_from(
        &mut self,
        e: &impl crate::KvDev,
        source: &Self,
        rows: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.basis.validate_copy(source.basis)?;
        self.validate()?;
        source.validate()?;
        if self.width != source.width || rows > self.capacity || rows > source.capacity {
            return Err("NVFP4 latent prefix geometry mismatch".into());
        }
        source.error.check()?;
        self.error.check()?;
        let layout = LatentLayout::new(LatentFormat::Nvfp4, self.width)?;
        e.copy_u8_range_into(
            &mut self.payload,
            0,
            &source.payload,
            0,
            rows * layout.payload_bytes,
        )?;
        e.copy_u8_range_into(
            &mut self.scales,
            0,
            &source.scales,
            0,
            rows * layout.block_scale_bytes,
        )?;
        e.copy_range_into(&mut self.macros, 0, &source.macros, 0, rows)?;
        Ok(())
    }

    pub fn snapshot(
        &self,
        e: &impl crate::KvDev,
        rows: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if rows > self.capacity {
            return Err("NVFP4 latent snapshot exceeds capacity".into());
        }
        self.validate()?;
        let mut snapshot = Self::new_with_basis(e, self.width, rows, self.basis)?;
        snapshot.copy_prefix_from(e, self, rows)?;
        Ok(snapshot)
    }
}

fn encode_fp4(x: f32) -> u8 {
    let ax = x.abs();
    let mut best = 0;
    for (i, &value) in E2M1.iter().enumerate().skip(1) {
        let distance = (ax - value).abs();
        let previous = (ax - E2M1[best]).abs();
        if distance < previous || (distance == previous && i % 2 == 0) {
            best = i;
        }
    }
    best as u8 | if x.is_sign_negative() { 8 } else { 0 }
}

#[derive(Clone, Debug)]
pub struct PackedRow {
    pub payload: Vec<u8>,
    pub block_scales: Vec<u8>,
    pub macro_scale: f32,
}

struct EncodedBlock {
    payload: [u8; 8],
    scale: u8,
    sse: f64,
}

fn encode_block(xs: &[f32], macro_scale: f32, multiplier: f64) -> EncodedBlock {
    let peak = xs.iter().fold(0f32, |a, x| a.max(x.abs()));
    let ideal = ((peak as f64 * multiplier) / (6.0 * macro_scale as f64)) as f32;
    let scale = f32_to_fp8_e4m3(ideal);
    let scale_value = fp8_e4m3_to_f32(scale);
    let divisor = scale_value as f64 * macro_scale as f64;
    let mut block = EncodedBlock {
        payload: [0; 8],
        scale,
        sse: 0.,
    };
    for (i, &x) in xs.iter().enumerate() {
        let code = if divisor == 0. {
            0
        } else {
            encode_fp4((x as f64 / divisor) as f32)
        };
        block.payload[i / 2] |= code << ((i % 2) * 4);
        let value = E2M1[(code & 7) as usize] * if code & 8 == 0 { 1. } else { -1. };
        // Same two f32 multiplies as the unchanged CUDA gather/direct readers.
        let reconstructed = (value * scale_value) * macro_scale;
        let difference = reconstructed as f64 - x as f64;
        let squared = difference * difference;
        block.sse += squared;
    }
    block
}

impl PackedRow {
    pub fn encode(values: &[f32]) -> Result<Self, &'static str> {
        let layout = LatentLayout::new(LatentFormat::Nvfp4, values.len())?;
        if values.iter().any(|x| !x.is_finite()) {
            return Err("latent row contains nonfinite values");
        }
        let amax = values.iter().fold(0f32, |a, x| a.max(x.abs()));
        // Never let a new row change the scale of previously appended history.
        // f64 intermediates avoid overflowing reciprocals for very small rows.
        let macro_scale = if amax == 0.0 {
            1.0
        } else {
            ((amax as f64 / (6.0 * 448.0)) as f32).max(f32::from_bits(1))
        };
        let (mut row, mut best_sse) = Self::candidate(values, macro_scale, false, layout);
        let alternative_macro = if amax == 0. {
            1.
        } else {
            ((amax as f64 / 1536.) as f32).max(f32::from_bits(1))
        };
        // Legacy comes first, then MSE with its macro, then MSE with M=4 headroom.
        // Strict comparison preserves the exact old bytes whenever the row scores tie.
        for candidate_macro in [macro_scale, alternative_macro] {
            let (candidate, sse) = Self::candidate(values, candidate_macro, true, layout);
            if sse < best_sse {
                row = candidate;
                best_sse = sse;
            }
        }
        if !best_sse.is_finite() {
            return Err("latent reconstruction overflow");
        }
        Ok(row)
    }

    fn candidate(
        values: &[f32],
        macro_scale: f32,
        adaptive: bool,
        layout: LatentLayout,
    ) -> (Self, f64) {
        let mut row = Self {
            payload: vec![0; layout.payload_bytes],
            block_scales: vec![0; layout.block_scale_bytes],
            macro_scale,
        };
        // Mirror CUDA's 256-thread block-stride accumulation and reduction exactly.
        // f64 products/additions are separate (not fused) on both implementations.
        let mut totals = [0f64; 256];
        for (block, xs) in values.chunks_exact(16).enumerate() {
            let mut encoded = encode_block(xs, macro_scale, 1.);
            if adaptive {
                let m4 = encode_block(xs, macro_scale, 1.5);
                if m4.sse < encoded.sse {
                    encoded = m4;
                }
            }
            totals[block % 256] += encoded.sse;
            row.block_scales[block] = encoded.scale;
            row.payload[block * 8..(block + 1) * 8].copy_from_slice(&encoded.payload);
        }
        for offset in [128, 64, 32, 16, 8, 4, 2, 1] {
            for i in 0..offset {
                totals[i] += totals[i + offset];
            }
        }
        (row, totals[0])
    }

    pub fn decode(&self) -> Result<Vec<f32>, &'static str> {
        let width = self
            .payload
            .len()
            .checked_mul(2)
            .ok_or("latent width overflow")?;
        let layout = LatentLayout::new(LatentFormat::Nvfp4, width)?;
        if self.block_scales.len() != layout.block_scale_bytes
            || !self.macro_scale.is_finite()
            || self.macro_scale <= 0.0
            || self.block_scales.iter().any(|s| *s > 0x7e)
        {
            return Err("invalid latent NVFP4 scales");
        }
        let mut values = Vec::with_capacity(width);
        for (i, byte) in self.payload.iter().enumerate() {
            let scale = fp8_e4m3_to_f32(self.block_scales[i / 8]) as f64 * self.macro_scale as f64;
            for code in [byte & 15, byte >> 4] {
                let sign = if code & 8 != 0 { -1.0 } else { 1.0 };
                let value = (E2M1[(code & 7) as usize] as f64 * scale * sign) as f32;
                if !value.is_finite() {
                    return Err("latent reconstruction overflow");
                }
                values.push(value);
            }
        }
        Ok(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Frozen pre-selection encoder, independent of the candidate-selection code.
    fn legacy(values: &[f32]) -> PackedRow {
        let peak = values.iter().fold(0f32, |a, x| a.max(x.abs()));
        let macro_scale = if peak == 0. {
            1.
        } else {
            ((peak as f64 / 2688.) as f32).max(f32::from_bits(1))
        };
        let mut row = PackedRow {
            payload: Vec::new(),
            block_scales: Vec::new(),
            macro_scale,
        };
        for block in values.chunks_exact(16) {
            let peak = block.iter().fold(0f32, |a, x| a.max(x.abs()));
            let scale = f32_to_fp8_e4m3((peak as f64 / (6. * macro_scale as f64)) as f32);
            row.block_scales.push(scale);
            let divisor = fp8_e4m3_to_f32(scale) as f64 * macro_scale as f64;
            for pair in block.chunks_exact(2) {
                let code = |x: f32| {
                    if divisor == 0. {
                        0
                    } else {
                        encode_fp4((x as f64 / divisor) as f32)
                    }
                };
                row.payload.push(code(pair[0]) | (code(pair[1]) << 4));
            }
        }
        row
    }

    fn sse(input: &[f32], row: &PackedRow) -> f64 {
        input
            .iter()
            .zip(row.decode().unwrap())
            .map(|(x, y)| {
                let d = *x as f64 - y as f64;
                d * d
            })
            .sum()
    }

    #[test]
    fn mse_peak_block_gets_m4_headroom() {
        let input: Vec<f32> = (0..512)
            .map(|i| if i % 16 == 0 { 1. } else { 0.75 })
            .collect();
        let old = legacy(&input);
        assert!(sse(&input, &old) > 3.);
        let row = PackedRow::encode(&input).unwrap();
        assert_eq!(row.macro_scale.to_bits(), (1f32 / 1536.).to_bits());
        assert!(row.block_scales.iter().all(|s| fp8_e4m3_to_f32(*s) == 384.));
        assert_eq!(sse(&input, &row), 0.);
    }

    #[test]
    fn mse_ties_keep_existing_bytes() {
        for input in [
            vec![0.; 512],
            vec![-0.; 512],
            vec![1.; 512],
            (0..512)
                .map(|i| if i % 16 == 0 { 1. } else { 0. })
                .collect(),
        ] {
            let old = legacy(&input);
            let row = PackedRow::encode(&input).unwrap();
            assert_eq!(row.payload, old.payload);
            assert_eq!(row.block_scales, old.block_scales);
            assert_eq!(row.macro_scale.to_bits(), old.macro_scale.to_bits());
        }
    }

    #[test]
    fn mse_never_worsens_synthetic_rows_including_extremes() {
        let mut state = 7u32;
        for amplitude in [
            f32::from_bits(1),
            f32::MIN_POSITIVE,
            1e-20,
            1.,
            1e20,
            f32::MAX,
        ] {
            for _ in 0..16 {
                let input: Vec<f32> = (0..512)
                    .map(|i| {
                        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                        if i == 0 {
                            amplitude
                        } else {
                            let magnitude = ((state >> 8) as f32 / 16777216.) * amplitude;
                            if state & 1 == 0 {
                                magnitude
                            } else {
                                -magnitude
                            }
                        }
                    })
                    .collect();
                let old = legacy(&input);
                let row = PackedRow::encode(&input).unwrap();
                assert!(
                    sse(&input, &row) <= sse(&input, &old),
                    "amplitude {amplitude}"
                );
                let again = PackedRow::encode(&input).unwrap();
                assert_eq!(row.payload, again.payload);
                assert_eq!(row.block_scales, again.block_scales);
                assert_eq!(row.macro_scale.to_bits(), again.macro_scale.to_bits());
            }
        }
    }

    #[test]
    fn mse_block_stride_reduction_keeps_legacy_candidate() {
        // More than 256 blocks exercises the per-lane strided accumulation,
        // rather than only the one-block-per-active-thread rank512 shape.
        let input: Vec<f32> = (0..16 * 513)
            .map(|i| ((i * 37 % 1021) as f32 - 510.) / 73.)
            .collect();
        let old = legacy(&input);
        let row = PackedRow::encode(&input).unwrap();
        assert!(sse(&input, &row) <= sse(&input, &old));
        assert_eq!(row.payload.len(), old.payload.len());
        assert_eq!(row.block_scales.len(), old.block_scales.len());
    }

    #[test]
    fn fp4_grid_and_even_ties() {
        for (i, &v) in E2M1.iter().enumerate() {
            assert_eq!(encode_fp4(v), i as u8);
            assert_eq!(encode_fp4(-v), i as u8 | 8);
        }
        for i in 0..7 {
            assert_eq!(
                encode_fp4((E2M1[i] + E2M1[i + 1]) / 2.0),
                (if i % 2 == 0 { i } else { i + 1 }) as u8
            );
        }
    }

    #[test]
    fn sequential_nibbles_and_scale_reconstruction() {
        let row = PackedRow {
            payload: vec![0x21; 8],
            block_scales: vec![0x38],
            macro_scale: 2.0,
        };
        assert_eq!(row.decode().unwrap(), [1.0, 2.0].repeat(8));
    }

    #[test]
    fn zeros_and_block_ranges() {
        assert_eq!(
            PackedRow::encode(&[0.0; 512]).unwrap().decode().unwrap(),
            vec![0.0; 512]
        );
        let mut input = vec![6.0; 16];
        input.extend([-0.5; 16]);
        let row = PackedRow::encode(&input).unwrap();
        assert!(row.macro_scale.is_finite() && row.macro_scale > 0.);
        assert_eq!(row.block_scales.len(), input.len() / 16);
        assert!(row.block_scales.iter().all(|s| {
            *s <= 0x7e && fp8_e4m3_to_f32(*s).is_finite() && fp8_e4m3_to_f32(*s) > 0.
        }));
        assert!(sse(&input, &row) <= sse(&input, &legacy(&input)));
        for (x, y) in input.iter().zip(row.decode().unwrap()) {
            assert!(y.is_finite());
            assert!((x - y).abs() < 0.04);
        }
    }

    #[test]
    fn invalid_inputs_refuse() {
        assert!(PackedRow::encode(&[1.0; 15]).is_err());
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(PackedRow::encode(&[bad; 16]).is_err());
        }
        let mut row = PackedRow::encode(&[1.0; 16]).unwrap();
        row.block_scales[0] = 0x7f;
        assert!(row.decode().is_err());
        row.block_scales[0] = 0x38;
        row.macro_scale = 0.0;
        assert!(row.decode().is_err());
    }
}
