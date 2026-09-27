//! Qwen2VLImageProcessor patch rows for already prepared MiMo image or video frames.
//!
//! The pinned `preprocessor_config.json` selects patch size 16, temporal patch
//! size 2, and spatial merge size 2. This follows the reshape and transpose in
//! Transformers 5.3 `image_processing_qwen2_vl.py::_preprocess`. Decoding,
//! resizing, rescaling, and normalization happen before this component.

use memra_gguf::model_packs::mimo_v2::vision::MiMoVisionGrid;

const CHANNELS: usize = 3;
const TEMPORAL_PATCH: usize = 2;
const SPATIAL_PATCH: usize = 16;
const MERGE: usize = 2;
const SMART_RESIZE_FACTOR: usize = SPATIAL_PATCH * MERGE;
const SMART_RESIZE_MIN_PIXELS: usize = 3_136;
const SMART_RESIZE_MAX_PIXELS: usize = 12_845_056;
const MAX_FRAME_GROUPS: usize = 32;
const MAX_PATCHES: usize = 256;

pub const MIMO_PATCH_ROW_WIDTH: usize = CHANNELS * TEMPORAL_PATCH * SPATIAL_PATCH * SPATIAL_PATCH;

/// Source Qwen2VL resize dimensions and the Conv3D patch grid for one temporal group.
///
/// This describes geometry only. In particular, `patches_per_group` may exceed
/// the current native GPU projector limit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MiMoSmartResizeGeometry {
    pub resized_height: usize,
    pub resized_width: usize,
    pub grid_height: usize,
    pub grid_width: usize,
    pub patches_per_group: usize,
}

fn round_to_factor_ties_even(value: usize) -> Result<usize, String> {
    let quotient = value / SMART_RESIZE_FACTOR;
    let remainder = value % SMART_RESIZE_FACTOR;
    let round_up = remainder > SMART_RESIZE_FACTOR / 2
        || (remainder == SMART_RESIZE_FACTOR / 2 && quotient % 2 == 1);
    quotient
        .checked_add(usize::from(round_up))
        .and_then(|rounded| rounded.checked_mul(SMART_RESIZE_FACTOR))
        .ok_or_else(|| "MiMo smart-resize rounded dimension overflows".into())
}

fn checked_scaled_multiple(units: f64) -> Result<usize, String> {
    if !units.is_finite() || units < 0.0 || units >= (usize::MAX / SMART_RESIZE_FACTOR) as f64 {
        return Err("MiMo smart-resize scaled dimension overflows".into());
    }
    (units as usize)
        .checked_mul(SMART_RESIZE_FACTOR)
        .ok_or_else(|| "MiMo smart-resize scaled dimension overflows".into())
}

/// Match Transformers 5.3 `Qwen2VLImageProcessor` `smart_resize` using the
/// pinned MiMo patch size 16, merge size 2, and pixel limits 3136..=12845056.
/// The pinned `preprocessor_config.json` SHA-256 is
/// `b269e51bdc1c53ef7c82522984378513f462869cf83c4c75ec3f98bd92aa7972`.
///
/// The initial nearest-factor rounding uses Python's ties-to-even rule. Scaling
/// uses f64 square roots and the source's operation order. This does not resize
/// image pixels or limit the result to the native GPU projector's 256 patches.
pub fn mimo_smart_resize_geometry(
    height: usize,
    width: usize,
) -> Result<MiMoSmartResizeGeometry, String> {
    if height == 0 || width == 0 {
        return Err("MiMo smart-resize height and width must be positive".into());
    }
    let aspect_ratio = height.max(width) as f64 / height.min(width) as f64;
    if aspect_ratio > 200.0 {
        return Err("MiMo smart-resize aspect ratio exceeds 200".into());
    }
    let original_pixels = height
        .checked_mul(width)
        .ok_or("MiMo smart-resize source pixel count overflows")?;
    let mut resized_height = round_to_factor_ties_even(height)?;
    let mut resized_width = round_to_factor_ties_even(width)?;
    let rounded_pixels = resized_height
        .checked_mul(resized_width)
        .ok_or("MiMo smart-resize rounded pixel count overflows")?;
    if rounded_pixels > SMART_RESIZE_MAX_PIXELS {
        let beta = (original_pixels as f64 / SMART_RESIZE_MAX_PIXELS as f64).sqrt();
        resized_height = checked_scaled_multiple(
            (height as f64 / beta / SMART_RESIZE_FACTOR as f64)
                .floor()
                .max(1.0),
        )?;
        resized_width = checked_scaled_multiple(
            (width as f64 / beta / SMART_RESIZE_FACTOR as f64)
                .floor()
                .max(1.0),
        )?;
    } else if rounded_pixels < SMART_RESIZE_MIN_PIXELS {
        let beta = (SMART_RESIZE_MIN_PIXELS as f64 / original_pixels as f64).sqrt();
        resized_height =
            checked_scaled_multiple((height as f64 * beta / SMART_RESIZE_FACTOR as f64).ceil())?;
        resized_width =
            checked_scaled_multiple((width as f64 * beta / SMART_RESIZE_FACTOR as f64).ceil())?;
    }
    let grid_height = resized_height / SPATIAL_PATCH;
    let grid_width = resized_width / SPATIAL_PATCH;
    let patches_per_group = grid_height
        .checked_mul(grid_width)
        .ok_or("MiMo smart-resize patch grid overflows")?;
    Ok(MiMoSmartResizeGeometry {
        resized_height,
        resized_width,
        grid_height,
        grid_width,
        patches_per_group,
    })
}

/// Admit source geometry to the current native GPU projector for a requested
/// number of temporal groups. Returns its patch grid only when all caps pass.
pub fn admit_mimo_gpu_projector_grid(
    geometry: MiMoSmartResizeGeometry,
    temporal_groups: usize,
) -> Result<MiMoVisionGrid, String> {
    if temporal_groups == 0 || temporal_groups > MAX_FRAME_GROUPS {
        return Err("MiMo GPU projector needs 1..=32 temporal groups".into());
    }
    if geometry.resized_height == 0
        || geometry.resized_width == 0
        || !geometry.resized_height.is_multiple_of(SMART_RESIZE_FACTOR)
        || !geometry.resized_width.is_multiple_of(SMART_RESIZE_FACTOR)
        || geometry.grid_height != geometry.resized_height / SPATIAL_PATCH
        || geometry.grid_width != geometry.resized_width / SPATIAL_PATCH
        || geometry.grid_height.checked_mul(geometry.grid_width) != Some(geometry.patches_per_group)
    {
        return Err("MiMo GPU projector grid is invalid".into());
    }
    let patch_count = geometry
        .patches_per_group
        .checked_mul(temporal_groups)
        .ok_or("MiMo GPU projector patch count overflows")?;
    if patch_count > MAX_PATCHES {
        return Err("MiMo GPU projector exceeds 256 patches".into());
    }
    Ok(MiMoVisionGrid {
        frames: u32::try_from(temporal_groups).map_err(|_| "MiMo temporal grid overflows u32")?,
        height: u32::try_from(geometry.grid_height)
            .map_err(|_| "MiMo height grid overflows u32")?,
        width: u32::try_from(geometry.grid_width).map_err(|_| "MiMo width grid overflows u32")?,
    })
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
            frames: u32::try_from(grid_t).map_err(|_| "MiMo temporal grid overflows u32")?,
            height: u32::try_from(grid_h).map_err(|_| "MiMo height grid overflows u32")?,
            width: u32::try_from(grid_w).map_err(|_| "MiMo width grid overflows u32")?,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_uses_python_ties_to_even_in_both_dimensions() {
        // 80/32 is 2.5, 112/32 is 3.5, and 144/32 is 4.5.
        assert_eq!(
            mimo_smart_resize_geometry(80, 128).unwrap(),
            MiMoSmartResizeGeometry {
                resized_height: 64,
                resized_width: 128,
                grid_height: 4,
                grid_width: 8,
                patches_per_group: 32,
            }
        );
        assert_eq!(
            mimo_smart_resize_geometry(112, 128).unwrap(),
            MiMoSmartResizeGeometry {
                resized_height: 128,
                resized_width: 128,
                grid_height: 8,
                grid_width: 8,
                patches_per_group: 64,
            }
        );
        assert_eq!(
            mimo_smart_resize_geometry(128, 144).unwrap().resized_width,
            128
        );
    }

    #[test]
    fn smart_resize_preserves_pinned_min_max_and_source_grid() {
        let minimum = mimo_smart_resize_geometry(32, 32).unwrap();
        assert_eq!((minimum.resized_height, minimum.resized_width), (64, 64));
        assert_eq!(
            (
                minimum.grid_height,
                minimum.grid_width,
                minimum.patches_per_group,
            ),
            (4, 4, 16)
        );
        let narrow = mimo_smart_resize_geometry(16, 64).unwrap();
        assert_eq!((narrow.resized_height, narrow.resized_width), (32, 128));
        assert_eq!(
            (
                narrow.grid_height,
                narrow.grid_width,
                narrow.patches_per_group,
            ),
            (2, 8, 16)
        );
        // The source chooses a branch from the rounded area, then computes
        // beta from the original area.
        let rounded_below_minimum = mimo_smart_resize_geometry(15, 210).unwrap();
        assert_eq!(
            (
                rounded_below_minimum.resized_height,
                rounded_below_minimum.resized_width,
            ),
            (32, 224)
        );
        let maximum = mimo_smart_resize_geometry(4096, 4096).unwrap();
        assert_eq!(
            (
                maximum.resized_height,
                maximum.resized_width,
                maximum.grid_height,
                maximum.grid_width,
                maximum.patches_per_group,
            ),
            (3584, 3584, 224, 224, 50_176)
        );
        let rounded_above_maximum = mimo_smart_resize_geometry(3000, 4272).unwrap();
        assert_eq!(
            (
                rounded_above_maximum.resized_height,
                rounded_above_maximum.resized_width,
            ),
            (2976, 4256)
        );
        let unchanged = mimo_smart_resize_geometry(64, 96).unwrap();
        assert_eq!(
            (
                unchanged.resized_height,
                unchanged.resized_width,
                unchanged.grid_height,
                unchanged.grid_width,
                unchanged.patches_per_group,
            ),
            (64, 96, 4, 6, 24)
        );
    }

    #[test]
    fn smart_resize_rejects_invalid_extents_and_aspects() {
        assert!(mimo_smart_resize_geometry(0, 32).is_err());
        assert!(mimo_smart_resize_geometry(32, 0).is_err());
        let aspect_edge = mimo_smart_resize_geometry(1, 200).unwrap();
        assert_eq!(
            (aspect_edge.resized_height, aspect_edge.resized_width),
            (32, 800)
        );
        assert!(
            mimo_smart_resize_geometry(1, 201)
                .unwrap_err()
                .contains("aspect ratio")
        );
        assert!(
            mimo_smart_resize_geometry(usize::MAX / 2, usize::MAX / 2)
                .unwrap_err()
                .contains("source pixel count overflows")
        );
    }

    #[test]
    fn projector_admission_is_separate_and_fails_closed_above_256_patches() {
        let boundary = mimo_smart_resize_geometry(256, 256).unwrap();
        assert_eq!(boundary.patches_per_group, 256);
        assert_eq!(
            admit_mimo_gpu_projector_grid(boundary, 1).unwrap(),
            MiMoVisionGrid {
                frames: 1,
                height: 16,
                width: 16,
            }
        );
        let source_only = mimo_smart_resize_geometry(512, 512).unwrap();
        assert_eq!(
            (
                source_only.resized_height,
                source_only.resized_width,
                source_only.patches_per_group,
            ),
            (512, 512, 1024)
        );
        assert!(
            admit_mimo_gpu_projector_grid(source_only, 1)
                .unwrap_err()
                .contains("256 patches")
        );
        let small = mimo_smart_resize_geometry(32, 32).unwrap();
        assert_eq!(admit_mimo_gpu_projector_grid(small, 16).unwrap().frames, 16);
        assert!(
            admit_mimo_gpu_projector_grid(small, 17)
                .unwrap_err()
                .contains("256 patches")
        );
        assert!(admit_mimo_gpu_projector_grid(small, 0).is_err());
        assert!(admit_mimo_gpu_projector_grid(small, 33).is_err());
        let forged = MiMoSmartResizeGeometry {
            patches_per_group: 1,
            ..source_only
        };
        assert!(admit_mimo_gpu_projector_grid(forged, 1).is_err());
    }

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
