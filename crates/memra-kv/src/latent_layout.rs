//! Checked geometry for append-only latent history. No runtime dispatch is enabled here.
//! NVFP4 rows use sequential low/high nibbles, one E4M3 scale per 16 values,
//! and one f32 macro scale per row. Row-local macro scales keep earlier rows
//! immutable when later tokens have a different range. Index/recurrent planes
//! are separate and must not be counted as compressed by this layout.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LatentFormat {
    F32,
    Nvfp4,
}

/// Experiment-only selection, shared by allocation and admission. This is a
/// NoPE DSA latent-plane program, not a generic full-attention KV dtype switch.
pub fn nvfp4_enabled() -> bool {
    std::env::var("MEMRA_GLM53_NVFP4_LATENT").as_deref() == Ok("1")
}

pub fn selected_layout(width: usize, index_width: usize) -> Result<LatentLayout, &'static str> {
    layout_for_selection(nvfp4_enabled(), width, index_width)
}

pub fn layout_for_selection(
    enabled: bool,
    width: usize,
    index_width: usize,
) -> Result<LatentLayout, &'static str> {
    if enabled {
        if width != 512 || index_width == 0 {
            return Err("NVFP4 latent storage requires 512-wide NoPE DSA history");
        }
        LatentLayout::new(LatentFormat::Nvfp4, width)
    } else {
        LatentLayout::new(LatentFormat::F32, width)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LatentLayout {
    pub format: LatentFormat,
    pub width: usize,
    pub payload_bytes: usize,
    pub block_scale_bytes: usize,
    pub row_scale_bytes: usize,
}

impl LatentLayout {
    pub fn new(format: LatentFormat, width: usize) -> Result<Self, &'static str> {
        if width == 0 {
            return Err("latent width must be nonzero");
        }
        let (payload_bytes, block_scale_bytes, row_scale_bytes) = match format {
            LatentFormat::F32 => (width.checked_mul(4).ok_or("latent width overflow")?, 0, 0),
            LatentFormat::Nvfp4 => {
                if !width.is_multiple_of(16) {
                    return Err("NVFP4 latent width must be divisible by 16");
                }
                (width / 2, width / 16, 4)
            }
        };
        let layout = Self {
            format,
            width,
            payload_bytes,
            block_scale_bytes,
            row_scale_bytes,
        };
        layout.allocation_bytes(1)?;
        Ok(layout)
    }

    /// Storage planes only. Callers must additionally account for index state,
    /// device counters, workspace, prefix copies and draft caches.
    pub fn allocation_bytes(self, rows: usize) -> Result<usize, &'static str> {
        self.payload_bytes
            .checked_add(self.block_scale_bytes)
            .and_then(|n| n.checked_add(self.row_scale_bytes))
            .and_then(|n| n.checked_mul(rows))
            .ok_or("latent allocation overflow")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvfp4_counts_both_scale_planes() {
        let layout = LatentLayout::new(LatentFormat::Nvfp4, 512).unwrap();
        assert_eq!(
            (
                layout.payload_bytes,
                layout.block_scale_bytes,
                layout.row_scale_bytes
            ),
            (256, 32, 4)
        );
        assert_eq!(layout.allocation_bytes(218_000), Ok(63_656_000));
        assert_eq!(
            LatentLayout::new(LatentFormat::F32, 512)
                .unwrap()
                .allocation_bytes(218_000),
            Ok(446_464_000)
        );
    }

    #[test]
    fn refuses_invalid_widths_and_overflow() {
        for format in [LatentFormat::F32, LatentFormat::Nvfp4] {
            assert!(LatentLayout::new(format, 0).is_err());
            assert!(
                LatentLayout::new(format, 512)
                    .unwrap()
                    .allocation_bytes(usize::MAX)
                    .is_err()
            );
        }
        assert!(LatentLayout::new(LatentFormat::Nvfp4, 511).is_err());
        assert!(LatentLayout::new(LatentFormat::F32, usize::MAX).is_err());
    }

    #[test]
    fn empty_history_has_no_plane_storage() {
        for format in [LatentFormat::F32, LatentFormat::Nvfp4] {
            assert_eq!(
                LatentLayout::new(format, 576).unwrap().allocation_bytes(0),
                Ok(0)
            );
        }
    }
}
