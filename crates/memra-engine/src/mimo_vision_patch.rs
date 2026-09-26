//! Pinned MiMo Conv3D patch projection after pixel patchification.
//! This bounded BF16 path is a visual component oracle, not an image route.

use crate::Engine;
use crate::model::GpuTensor;
use cudarc::driver::CudaSlice;

type Fail = Box<dyn std::error::Error>;

const INPUT: usize = 3 * 2 * 16 * 16;
const OUTPUT: usize = 1280;
const MAX_PATCHES: usize = 256;

fn validate_projection(
    patches: usize,
    pixel_elements: usize,
    weight_ne: &[u64],
) -> Result<(), &'static str> {
    if !(1..=MAX_PATCHES).contains(&patches) {
        return Err("MiMo vision patch projection is outside 1..=256 patches");
    }
    if patches.checked_mul(INPUT) != Some(pixel_elements)
        || weight_ne != [INPUT as u64, OUTPUT as u64]
    {
        return Err("MiMo vision Conv3D patch operand geometry changed");
    }
    Ok(())
}

impl Engine {
    /// Project already patchified `[patches, 3, 2, 16, 16]` f32 pixels through
    /// the pinned BF16 Conv3D operand and return BF16-rounded f32 rows. The
    /// source processor owns resize, normalization, frame pairing, and merge
    /// unit ordering before this call.
    pub fn mimo_vision_patch_project(
        &self,
        weight: &GpuTensor,
        pixels: &CudaSlice<f32>,
        patches: usize,
    ) -> Result<CudaSlice<f32>, Fail> {
        let GpuTensor::FloatBf16 { data, ne } = weight else {
            return Err("MiMo vision patch projection requires source BF16 weights".into());
        };
        validate_projection(patches, pixels.len(), ne)?;
        self.gpu.ctx.bind_to_thread()?;
        let ordinal = self.stream().context().ordinal();
        if pixels.ordinal() != ordinal || data.ordinal() != ordinal {
            return Err("MiMo vision patch projection crossed GPU devices".into());
        }
        // PyTorch Conv3D casts the pixel operand to its BF16 weight dtype and
        // stores BF16 output. Retain those two rounding boundaries explicitly.
        let input_bf16 = self.f32_to_bf16(pixels, patches * INPUT)?;
        let rounded_input = self.bf16_to_f32(&input_bf16.slice(..), patches * INPUT)?;
        let projected = self.matmul(weight, &rounded_input, patches)?;
        if projected.len() != patches * OUTPUT {
            return Err("MiMo vision patch projection returned wrong extent".into());
        }
        let output_bf16 = self.f32_to_bf16(&projected, patches * OUTPUT)?;
        self.bf16_to_f32(&output_bf16.slice(..), patches * OUTPUT)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_operand_requires_pinned_conv3d_extent() {
        assert!(validate_projection(1, INPUT, &[INPUT as u64, OUTPUT as u64]).is_ok());
        assert!(validate_projection(256, 256 * INPUT, &[INPUT as u64, OUTPUT as u64]).is_ok());
        assert!(validate_projection(0, 0, &[INPUT as u64, OUTPUT as u64]).is_err());
        assert!(validate_projection(257, 257 * INPUT, &[INPUT as u64, OUTPUT as u64]).is_err());
        assert!(validate_projection(1, INPUT - 1, &[INPUT as u64, OUTPUT as u64]).is_err());
        assert!(validate_projection(1, INPUT, &[OUTPUT as u64, INPUT as u64]).is_err());
    }

    #[test]
    #[ignore = "requires a dedicated MiMo GPU component lane"]
    fn gpu_patch_projection_preserves_conv3d_flatten_and_bf16_rounding() -> Result<(), Fail> {
        let gpu: usize = std::env::var("MEMRA_MIMO_COMPONENT_GPU")
            .unwrap_or_else(|_| "0".into())
            .parse()?;
        let engine = Engine::new(gpu)?;
        let mut weights = vec![0u8; INPUT * OUTPUT * 2];
        weights[0..2].copy_from_slice(&0x3f80u16.to_le_bytes());
        let last = (INPUT + INPUT - 1) * 2;
        weights[last..last + 2].copy_from_slice(&0x4000u16.to_le_bytes());
        let weight = GpuTensor::FloatBf16 {
            data: engine.htod_bytes(&weights)?,
            ne: vec![INPUT as u64, OUTPUT as u64],
        };
        let mut pixels = vec![0.0f32; 2 * INPUT];
        pixels[0] = 1.5;
        pixels[INPUT - 1] = 0.25;
        pixels[INPUT] = -2.0;
        pixels[2 * INPUT - 1] = 3.0;
        let projected = engine.mimo_vision_patch_project(&weight, &engine.htod(&pixels)?, 2)?;
        let projected = engine.dtoh(&projected)?;
        assert_eq!(projected.len(), 2 * OUTPUT);
        assert_eq!(projected[0].to_bits(), 1.5f32.to_bits());
        assert_eq!(projected[1].to_bits(), 0.5f32.to_bits());
        assert_eq!(projected[OUTPUT].to_bits(), (-2.0f32).to_bits());
        assert_eq!(projected[OUTPUT + 1].to_bits(), 6.0f32.to_bits());
        assert!(projected[2..OUTPUT].iter().all(|value| *value == 0.0));
        assert!(projected[OUTPUT + 2..].iter().all(|value| *value == 0.0));
        Ok(())
    }
}
