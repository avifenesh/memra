//! Portable Conv1D frontend oracle for the pinned MiMo audio tokenizer.
//!
//! Input rows are `[mel_frames, channels]` and the checkpoint weight is
//! `[out_channels, in_channels, 3]`. The publisher runs BF16 convolution,
//! then default erf GELU, then a second BF16 convolution and GELU.

use crate::speech::encoder::gelu_erf;

pub const MAX_COMPONENT_MEL_FRAMES: usize = 1_024;

#[inline]
fn bf16_to_f32(value: u16) -> f32 {
    f32::from_bits(u32::from(value) << 16)
}

#[inline]
fn bf16_round(value: f32) -> f32 {
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    bf16_to_f32((bits.wrapping_add(rounding) >> 16) as u16)
}

pub fn output_frames(input_frames: usize, stride: usize) -> Result<usize, String> {
    if !(1..=MAX_COMPONENT_MEL_FRAMES).contains(&input_frames) || !matches!(stride, 1 | 2) {
        return Err("MiMo codec Conv1D frames or stride outside component bounds".into());
    }
    Ok(if stride == 1 {
        input_frames
    } else {
        input_frames.div_ceil(2)
    })
}

/// Small portable oracle. Each convolution accumulates in f32, adds its BF16
/// bias, rounds the convolution output to BF16, applies erf GELU, and rounds
/// that activation to BF16. The source uses zero padding of one on both ends.
pub fn conv1d_gelu_bf16(
    input: &[f32],
    frames: usize,
    in_channels: usize,
    out_channels: usize,
    stride: usize,
    weight: &[u16],
    bias: &[u16],
) -> Result<Vec<f32>, String> {
    let output_frames = output_frames(frames, stride)?;
    if in_channels == 0
        || out_channels == 0
        || in_channels > 1_024
        || out_channels > 1_024
        || input.len() != frames * in_channels
        || weight.len() != out_channels * in_channels * 3
        || bias.len() != out_channels
        || input.iter().any(|value| !value.is_finite())
        || weight.iter().any(|&value| !bf16_to_f32(value).is_finite())
        || bias.iter().any(|&value| !bf16_to_f32(value).is_finite())
    {
        return Err("MiMo codec Conv1D input, BF16 weight, or bias extent changed".into());
    }
    let mut output = vec![0.0f32; output_frames * out_channels];
    for row in 0..output_frames {
        for out_channel in 0..out_channels {
            let mut sum = 0.0f32;
            for in_channel in 0..in_channels {
                for tap in 0..3 {
                    let position = row * stride + tap;
                    if position == 0 || position > frames {
                        continue;
                    }
                    let sample = bf16_round(input[(position - 1) * in_channels + in_channel]);
                    let value =
                        bf16_to_f32(weight[(out_channel * in_channels + in_channel) * 3 + tap]);
                    sum = sample.mul_add(value, sum);
                }
            }
            let conv = bf16_round(sum + bf16_to_f32(bias[out_channel]));
            let activated = bf16_round(gelu_erf(conv));
            if !activated.is_finite() {
                return Err("MiMo codec Conv1D produced a non-finite BF16 activation".into());
            }
            output[row * out_channels + out_channel] = activated;
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(value: f32) -> u16 {
        (bf16_round(value).to_bits() >> 16) as u16
    }

    #[test]
    fn padding_and_channel_tap_order_follow_conv1d() {
        // One output channel, two input channels. Channel 0 reads left and
        // right; channel 1 reads only the center with a distinct coefficient.
        let input = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0];
        let weight = [
            bits(1.0),
            bits(0.0),
            bits(1.0),
            bits(0.0),
            bits(0.5),
            bits(0.0),
        ];
        let output = conv1d_gelu_bf16(&input, 3, 2, 1, 1, &weight, &[bits(0.0)]).unwrap();
        let raw = [7.0, 14.0, 17.0];
        for (got, value) in output.iter().zip(raw) {
            let expected = bf16_round(gelu_erf(bf16_round(value)));
            assert_eq!(*got, expected);
        }
    }

    #[test]
    fn stride_two_uses_ceil_length_and_bf16_stage_boundaries() {
        let input = [1.005, 2.0, 3.0, 4.0, 5.0];
        let weight = [bits(1.0), bits(0.0), bits(0.0)];
        let bias = [bits(0.125)];
        let output = conv1d_gelu_bf16(&input, 5, 1, 1, 2, &weight, &bias).unwrap();
        assert_eq!(output.len(), 3);
        for (got, value) in output.iter().zip([0.0, 2.0, 4.0]) {
            let conv = bf16_round(bf16_round(value) + 0.125);
            assert_eq!(*got, bf16_round(gelu_erf(conv)));
        }
    }

    #[test]
    fn extent_and_nonfinite_values_are_refused() {
        let weight = [bits(0.0); 3];
        let bias = [bits(0.0)];
        assert!(conv1d_gelu_bf16(&[1.0], 0, 1, 1, 1, &weight, &bias).is_err());
        assert!(conv1d_gelu_bf16(&[1.0], 1, 1, 1, 3, &weight, &bias).is_err());
        assert!(conv1d_gelu_bf16(&[1.0], 1, 1, 1, 1, &weight[..2], &bias).is_err());
        assert!(conv1d_gelu_bf16(&[f32::NAN], 1, 1, 1, 1, &weight, &bias).is_err());
        assert!(conv1d_gelu_bf16(&[1.0], 1, 1, 1, 1, &[0x7f80; 3], &bias).is_err());
    }
}
