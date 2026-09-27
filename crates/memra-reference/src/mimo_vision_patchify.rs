//! Qwen2VLImageProcessor patch rows for already prepared MiMo image or video frames.
//!
//! The pinned `preprocessor_config.json` selects patch size 16, temporal patch
//! size 2, and spatial merge size 2. This follows the reshape and transpose in
//! Transformers 5.3 `image_processing_qwen2_vl.py::_preprocess`. Decoding,
//! resizing, rescaling, and normalization happen before this component.

const CHANNELS: usize = 3;
const TEMPORAL_PATCH: usize = 2;
const SPATIAL_PATCH: usize = 16;
const MERGE: usize = 2;
const MAX_FRAME_GROUPS: usize = 32;
const MAX_PATCHES: usize = 256;

pub const MIMO_PATCH_ROW_WIDTH: usize = CHANNELS * TEMPORAL_PATCH * SPATIAL_PATCH * SPATIAL_PATCH;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MiMoVisionGrid {
    /// Temporal groups after repeating the last frame when the input count is odd.
    pub frames: usize,
    /// Number of 16-pixel patches along the image height.
    pub height: usize,
    /// Number of 16-pixel patches along the image width.
    pub width: usize,
}

/// Convert contiguous, already resized, rescaled, normalized F32 frames in
/// channels-first `[T, 3, H, W]` order into row-major Conv3D input patches.
///
/// `shape_tchw` supplies all four input dimensions. The flat input must match
/// that shape exactly; dimensions are never inferred from its length. Rows are
/// ordered as `(grid_t, grid_h/2, grid_w/2, merge_h, merge_w)` and each row as
/// `(channel, temporal, patch_h, patch_w)`, matching the source transpose
/// `(0, 3, 6, 4, 7, 2, 1, 5, 8)`. Output shape is
/// `[grid_t * grid_h * grid_w, 3 * 2 * 16 * 16]`.
pub fn patchify_prepared_frames(
    frames: &[f32],
    shape_tchw: [usize; 4],
) -> Result<(Vec<f32>, MiMoVisionGrid), String> {
    let [frame_count, channels, height, width] = shape_tchw;
    if frame_count == 0 || channels != CHANNELS || height == 0 || width == 0 {
        return Err("MiMo prepared frames need positive T/H/W and exactly 3 channels".into());
    }
    if !height.is_multiple_of(SPATIAL_PATCH * MERGE) || !width.is_multiple_of(SPATIAL_PATCH * MERGE)
    {
        return Err("MiMo prepared frame height and width must be divisible by 32".into());
    }

    let grid_t = frame_count.div_ceil(TEMPORAL_PATCH);
    if grid_t > MAX_FRAME_GROUPS {
        return Err("MiMo prepared frames exceed 32 temporal groups".into());
    }
    let grid_h = height / SPATIAL_PATCH;
    let grid_w = width / SPATIAL_PATCH;
    let patch_count = grid_t
        .checked_mul(grid_h)
        .and_then(|count| count.checked_mul(grid_w))
        .ok_or("MiMo prepared patch count overflows")?;
    if patch_count > MAX_PATCHES {
        return Err("MiMo prepared frames exceed 256 GPU projector patches".into());
    }
    let expected_len = frame_count
        .checked_mul(channels)
        .and_then(|count| count.checked_mul(height))
        .and_then(|count| count.checked_mul(width))
        .ok_or("MiMo prepared frame extent overflows")?;
    if frames.len() != expected_len {
        return Err("MiMo prepared frame length differs from explicit TCHW shape".into());
    }
    if frames.iter().any(|value| !value.is_finite()) {
        return Err("MiMo prepared frames must contain only finite F32 values".into());
    }

    let mut patches = Vec::with_capacity(patch_count * MIMO_PATCH_ROW_WIDTH);
    for group in 0..grid_t {
        for block_h in 0..grid_h / MERGE {
            for block_w in 0..grid_w / MERGE {
                for merge_h in 0..MERGE {
                    for merge_w in 0..MERGE {
                        let pixel_h = (block_h * MERGE + merge_h) * SPATIAL_PATCH;
                        let pixel_w = (block_w * MERGE + merge_w) * SPATIAL_PATCH;
                        for channel in 0..CHANNELS {
                            for temporal in 0..TEMPORAL_PATCH {
                                let source_frame =
                                    (group * TEMPORAL_PATCH + temporal).min(frame_count - 1);
                                for patch_h in 0..SPATIAL_PATCH {
                                    let start = ((source_frame * CHANNELS + channel) * height
                                        + pixel_h
                                        + patch_h)
                                        * width
                                        + pixel_w;
                                    patches
                                        .extend_from_slice(&frames[start..start + SPATIAL_PATCH]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok((
        patches,
        MiMoVisionGrid {
            frames: grid_t,
            height: grid_h,
            width: grid_w,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coordinate(frame: usize, channel: usize, y: usize, x: usize) -> f32 {
        // Every axis has a separate, exactly representable digit range.
        (frame * 1_000_000
            + channel * 100_000
            + (y / 16) * 10_000
            + (x / 16) * 1_000
            + (y % 16) * 32
            + x % 16) as f32
    }

    fn prepared(t: usize, h: usize, w: usize) -> Vec<f32> {
        let mut input = Vec::with_capacity(t * CHANNELS * h * w);
        for frame in 0..t {
            for channel in 0..CHANNELS {
                for y in 0..h {
                    for x in 0..w {
                        input.push(coordinate(frame, channel, y, x));
                    }
                }
            }
        }
        input
    }

    #[test]
    fn coordinate_coded_rows_keep_merge_units_channels_time_and_pixels_in_source_order() {
        // Six merge units in a rectangular 4x6 patch grid. In particular,
        // row 4 starts the next merge unit at patch (0, 2), not patch (1, 0).
        const ROW_PATCHES: [(usize, usize); 24] = [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (0, 4),
            (0, 5),
            (1, 4),
            (1, 5),
            (2, 0),
            (2, 1),
            (3, 0),
            (3, 1),
            (2, 2),
            (2, 3),
            (3, 2),
            (3, 3),
            (2, 4),
            (2, 5),
            (3, 4),
            (3, 5),
        ];
        let input = prepared(2, 64, 96);
        let (rows, grid) = patchify_prepared_frames(&input, [2, 3, 64, 96]).unwrap();
        assert_eq!(
            grid,
            MiMoVisionGrid {
                frames: 1,
                height: 4,
                width: 6
            }
        );
        assert_eq!(rows.len(), ROW_PATCHES.len() * MIMO_PATCH_ROW_WIDTH);
        for (row, &(patch_h, patch_w)) in ROW_PATCHES.iter().enumerate() {
            for channel in 0..3 {
                for frame in 0..2 {
                    for y in 0..16 {
                        for x in 0..16 {
                            let col = ((channel * 2 + frame) * 16 + y) * 16 + x;
                            assert_eq!(
                                rows[row * MIMO_PATCH_ROW_WIDTH + col],
                                coordinate(frame, channel, patch_h * 16 + y, patch_w * 16 + x),
                                "row {row}, channel {channel}, frame {frame}, pixel ({y}, {x})"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn odd_tail_repeats_last_frame_in_every_patch_of_last_temporal_group() {
        let input = prepared(3, 32, 64);
        let (rows, grid) = patchify_prepared_frames(&input, [3, 3, 32, 64]).unwrap();
        assert_eq!(
            grid,
            MiMoVisionGrid {
                frames: 2,
                height: 2,
                width: 4
            }
        );
        assert_eq!(rows.len(), 16 * MIMO_PATCH_ROW_WIDTH);
        let row_patches = [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
        ];
        for (row, &(patch_h, patch_w)) in row_patches.iter().enumerate() {
            for channel in 0..3 {
                for temporal in 0..2 {
                    for y in 0..16 {
                        for x in 0..16 {
                            let col = ((channel * 2 + temporal) * 16 + y) * 16 + x;
                            assert_eq!(
                                rows[(8 + row) * MIMO_PATCH_ROW_WIDTH + col],
                                coordinate(2, channel, patch_h * 16 + y, patch_w * 16 + x),
                                "tail row {row}, channel {channel}, temporal {temporal}"
                            );
                        }
                    }
                }
            }
        }
        assert_eq!(rows[0], coordinate(0, 0, 0, 0));
        assert_eq!(rows[16 * 16], coordinate(1, 0, 0, 0));
    }

    #[test]
    fn single_image_repeats_into_both_temporal_slots() {
        let input = prepared(1, 32, 32);
        let (rows, grid) = patchify_prepared_frames(&input, [1, 3, 32, 32]).unwrap();
        assert_eq!(
            grid,
            MiMoVisionGrid {
                frames: 1,
                height: 2,
                width: 2
            }
        );
        for row in rows.chunks_exact(MIMO_PATCH_ROW_WIDTH) {
            for channel in 0..3 {
                let first = (channel * 2) * 16 * 16;
                let second = (channel * 2 + 1) * 16 * 16;
                assert_eq!(&row[first..first + 256], &row[second..second + 256]);
            }
        }
    }

    #[test]
    fn rejects_nonfinite_values_and_ambiguous_or_invalid_geometry() {
        let mut input = prepared(1, 32, 32);
        assert!(patchify_prepared_frames(&input, [0, 3, 32, 32]).is_err());
        assert!(patchify_prepared_frames(&input, [1, 4, 32, 32]).is_err());
        assert!(patchify_prepared_frames(&input, [1, 3, 0, 32]).is_err());
        assert!(patchify_prepared_frames(&input, [1, 3, 48, 32]).is_err());
        assert!(patchify_prepared_frames(&input, [1, 3, 32, 16]).is_err());
        assert!(patchify_prepared_frames(&input[1..], [1, 3, 32, 32]).is_err());
        input.push(0.0);
        assert!(patchify_prepared_frames(&input, [1, 3, 32, 32]).is_err());
        input.pop();
        input[0] = f32::NAN;
        assert!(patchify_prepared_frames(&input, [1, 3, 32, 32]).is_err());
        input[0] = f32::INFINITY;
        assert!(patchify_prepared_frames(&input, [1, 3, 32, 32]).is_err());
    }

    #[test]
    fn enforces_projector_patch_and_temporal_group_bounds() {
        let input = vec![0.0; 64 * 3 * 32 * 64];
        let (rows, grid) = patchify_prepared_frames(&input, [64, 3, 32, 64]).unwrap();
        assert_eq!(
            grid,
            MiMoVisionGrid {
                frames: 32,
                height: 2,
                width: 4
            }
        );
        assert_eq!(rows.len(), 256 * MIMO_PATCH_ROW_WIDTH);

        let too_many_groups = vec![0.0; 65 * 3 * 32 * 32];
        assert!(
            patchify_prepared_frames(&too_many_groups, [65, 3, 32, 32])
                .unwrap_err()
                .contains("32 temporal groups")
        );
        let too_many_patches = vec![0.0; 3 * 32 * 2080];
        assert!(
            patchify_prepared_frames(&too_many_patches, [1, 3, 32, 2080])
                .unwrap_err()
                .contains("256 GPU projector patches")
        );
    }
}
