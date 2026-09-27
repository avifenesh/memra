//! Bounded RGB8 pixel preparation for MiMo image and video patch inputs.
//!
//! The pinned `Qwen2VLImageProcessor` uses smart resize with factor 32, PIL
//! bicubic interpolation, rescale by 1/255, CLIP normalization, and channels
//! first F32. This module accepts decoded, contiguous RGB8 frames only. It
//! does not decode image or video files or assemble multimodal requests.

use crate::mimo_vision_patchify::{admit_mimo_gpu_projector_grid, mimo_smart_resize_geometry};

const CHANNELS: usize = 3;
const PRECISION_BITS: u32 = 22;
const COEFFICIENT_SCALE: f64 = (1_u64 << PRECISION_BITS) as f64;
const CLIP_MEAN: [f32; CHANNELS] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
const CLIP_STD: [f32; CHANNELS] = [0.268_629_54, 0.261_302_58, 0.275_777_11];

#[derive(Debug)]
struct Coefficients {
    first: usize,
    weights: Vec<i64>,
}

fn bicubic(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        ((1.5 * x - 2.5) * x * x) + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * -0.5
    } else {
        0.0
    }
}

// Pillow 12.1.1 Resample.c uses f64 filter weights and signed 22-bit fixed point
// coefficients for RGB8. Keep its center, clipping, normalization, and
// coefficient rounding order so both resizing passes round to bytes.
fn coefficients(input_size: usize, output_size: usize) -> Vec<Coefficients> {
    let scale = input_size as f64 / output_size as f64;
    let filter_scale = scale.max(1.0);
    let support = 2.0 * filter_scale;
    let inverse_filter_scale = 1.0 / filter_scale;
    (0..output_size)
        .map(|out| {
            let center = (out as f64 + 0.5) * scale;
            let first = ((center - support + 0.5) as usize).min(input_size);
            let end = ((center + support + 0.5) as usize).min(input_size);
            let mut weights = Vec::with_capacity(end - first);
            let mut total = 0.0;
            for x in first..end {
                let weight = bicubic((x as f64 - center + 0.5) * inverse_filter_scale);
                weights.push(weight);
                total += weight;
            }
            for weight in &mut weights {
                let normalized = if total != 0.0 {
                    *weight / total
                } else {
                    *weight
                };
                *weight = if normalized < 0.0 {
                    (-0.5 + normalized * COEFFICIENT_SCALE).trunc()
                } else {
                    (0.5 + normalized * COEFFICIENT_SCALE).trunc()
                };
            }
            Coefficients {
                first,
                weights: weights.into_iter().map(|weight| weight as i64).collect(),
            }
        })
        .collect()
}

fn rounded_byte(sum: i64) -> u8 {
    (sum >> PRECISION_BITS).clamp(0, 255) as u8
}

fn resize_rgb8(
    input: &[u8],
    input_height: usize,
    input_width: usize,
    output_height: usize,
    output_width: usize,
) -> Vec<u8> {
    let (horizontal, intermediate_width) = if input_width == output_width {
        (input.to_vec(), input_width)
    } else {
        let coeffs = coefficients(input_width, output_width);
        let mut pixels = vec![0_u8; input_height * output_width * CHANNELS];
        for y in 0..input_height {
            for (x, coeff) in coeffs.iter().enumerate() {
                for channel in 0..CHANNELS {
                    let mut sum = 1_i64 << (PRECISION_BITS - 1);
                    for (tap, &weight) in coeff.weights.iter().enumerate() {
                        let source = (y * input_width + coeff.first + tap) * CHANNELS + channel;
                        sum += i64::from(input[source]) * weight;
                    }
                    pixels[(y * output_width + x) * CHANNELS + channel] = rounded_byte(sum);
                }
            }
        }
        (pixels, output_width)
    };
    if input_height == output_height {
        return horizontal;
    }

    let coeffs = coefficients(input_height, output_height);
    let mut pixels = vec![0_u8; output_height * output_width * CHANNELS];
    for (y, coeff) in coeffs.iter().enumerate() {
        for x in 0..output_width {
            for channel in 0..CHANNELS {
                let mut sum = 1_i64 << (PRECISION_BITS - 1);
                for (tap, &weight) in coeff.weights.iter().enumerate() {
                    let source =
                        ((coeff.first + tap) * intermediate_width + x) * CHANNELS + channel;
                    sum += i64::from(horizontal[source]) * weight;
                }
                pixels[(y * output_width + x) * CHANNELS + channel] = rounded_byte(sum);
            }
        }
    }
    pixels
}

/// Prepare decoded RGB8 frames in contiguous `[T, H, W, 3]` order for
/// `patchify_prepared_frames`. Returns contiguous normalized F32 values and
/// their explicit `[T, 3, resized_H, resized_W]` shape.
///
/// Smart resize uses the pinned `preprocessor_config.json` (SHA-256
/// `b269e51bdc1c53ef7c82522984378513f462869cf83c4c75ec3f98bd92aa7972`).
/// The source's possible 12,845,056 pixels are narrowed to the current native
/// patch projector cap of 256 patches across at most 32 temporal groups before
/// any pixel buffer is allocated. Callers supply already decoded RGB bytes;
/// no color conversion, crop, timestamp selection, or file decoding is implied.
pub fn prepare_mimo_rgb8_frames(
    frames: &[u8],
    shape_thwc: [usize; 4],
) -> Result<(Vec<f32>, [usize; 4]), String> {
    let [frame_count, height, width, channels] = shape_thwc;
    if frame_count == 0 || height == 0 || width == 0 || channels != CHANNELS {
        return Err("MiMo RGB8 frames need positive T/H/W and exactly 3 channels".into());
    }
    let geometry = mimo_smart_resize_geometry(height, width)?;
    admit_mimo_gpu_projector_grid(geometry, frame_count.div_ceil(2))?;
    let frame_bytes = height
        .checked_mul(width)
        .and_then(|pixels| pixels.checked_mul(CHANNELS))
        .ok_or("MiMo RGB8 frame extent overflows")?;
    let expected_len = frame_count
        .checked_mul(frame_bytes)
        .ok_or("MiMo RGB8 frame sequence extent overflows")?;
    if frames.len() != expected_len {
        return Err("MiMo RGB8 frame length differs from explicit THWC shape".into());
    }

    let output_height = geometry.resized_height;
    let output_width = geometry.resized_width;
    let output_pixels = output_height * output_width;
    let mut prepared = vec![0.0_f32; frame_count * CHANNELS * output_pixels];
    for frame in 0..frame_count {
        let source = &frames[frame * frame_bytes..(frame + 1) * frame_bytes];
        let resized = if height == output_height && width == output_width {
            None
        } else {
            Some(resize_rgb8(
                source,
                height,
                width,
                output_height,
                output_width,
            ))
        };
        let rgb = resized.as_deref().unwrap_or(source);
        for channel in 0..CHANNELS {
            for pixel in 0..output_pixels {
                let scaled = (f64::from(rgb[pixel * CHANNELS + channel]) * (1.0 / 255.0)) as f32;
                prepared[(frame * CHANNELS + channel) * output_pixels + pixel] =
                    (scaled - CLIP_MEAN[channel]) / CLIP_STD[channel];
            }
        }
    }
    Ok((
        prepared,
        [frame_count, CHANNELS, output_height, output_width],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mimo_vision_patchify::patchify_prepared_frames;

    #[test]
    fn rgb8_to_prepared_channels_first_and_patch_rows() {
        let mut input = vec![0_u8; 2 * 64 * 96 * 3];
        for (index, value) in input.iter_mut().enumerate() {
            *value = (index % 256) as u8;
        }
        let (prepared, shape) = prepare_mimo_rgb8_frames(&input, [2, 64, 96, 3]).unwrap();
        assert_eq!(shape, [2, 3, 64, 96]);
        assert_eq!(prepared.len(), 2 * 3 * 64 * 96);
        assert!((prepared[0] - ((0.0 - CLIP_MEAN[0]) / CLIP_STD[0])).abs() < 1e-6);
        assert!((prepared[64 * 96] - ((1.0 / 255.0 - CLIP_MEAN[1]) / CLIP_STD[1])).abs() < 1e-6);
        let (patches, grid) = patchify_prepared_frames(&prepared, shape).unwrap();
        assert_eq!(patches.len(), 24 * 1536);
        assert_eq!((grid.frames, grid.height, grid.width), (1, 4, 6));
    }

    #[test]
    fn resize_and_odd_video_tail_reach_prepared_patcher() {
        let input = vec![137_u8; 3 * 33 * 65 * 3];
        let (prepared, shape) = prepare_mimo_rgb8_frames(&input, [3, 33, 65, 3]).unwrap();
        assert_eq!(shape, [3, 3, 64, 96]);
        let (patches, grid) = patchify_prepared_frames(&prepared, shape).unwrap();
        assert_eq!(patches.len(), 48 * 1536);
        assert_eq!((grid.frames, grid.height, grid.width), (2, 4, 6));
    }

    #[test]
    fn fails_closed_on_invalid_input_and_projector_caps() {
        assert!(prepare_mimo_rgb8_frames(&[], [0, 32, 32, 3]).is_err());
        assert!(prepare_mimo_rgb8_frames(&[], [1, 32, 32, 4]).is_err());
        assert!(prepare_mimo_rgb8_frames(&[], [1, 1, 201, 3]).is_err());
        assert!(
            prepare_mimo_rgb8_frames(&[], [1, 512, 512, 3])
                .unwrap_err()
                .contains("256 patches")
        );
        assert!(
            prepare_mimo_rgb8_frames(&[], [65, 32, 32, 3])
                .unwrap_err()
                .contains("32 temporal groups")
        );
        assert!(
            prepare_mimo_rgb8_frames(&[], [1, 32, 32, 3])
                .unwrap_err()
                .contains("length differs")
        );
        assert!(prepare_mimo_rgb8_frames(&[], [usize::MAX, 32, 32, 3]).is_err());
    }

    #[test]
    fn exactly_256_projector_patches_are_admitted() {
        let input = vec![73_u8; 270 * 270 * 3];
        let (prepared, shape) = prepare_mimo_rgb8_frames(&input, [1, 270, 270, 3]).unwrap();
        assert_eq!(shape, [1, 3, 256, 256]);
        let (patches, grid) = patchify_prepared_frames(&prepared, shape).unwrap();
        assert_eq!(patches.len(), 256 * 1536);
        assert_eq!((grid.frames, grid.height, grid.width), (1, 16, 16));
    }
}
