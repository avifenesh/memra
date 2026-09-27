//! Portable one-layer oracle for the pinned MiMo bundled audio encoder.
//!
//! The public helper accepts small dimensions for semantic tests. The resident
//! GPU caller binds the pinned 1024/16/4096 shape separately.

use crate::speech::encoder::gelu_erf;

pub const MAX_COMPONENT_TOKENS: usize = 256;

pub fn bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    f32::from_bits((bits.wrapping_add(rounding) >> 16) << 16)
}

pub fn decode(values: &[u16]) -> Vec<f32> {
    values
        .iter()
        .map(|&value| f32::from_bits(u32::from(value) << 16))
        .collect()
}

pub struct Norm<'a> {
    pub weight: &'a [u16],
    pub bias: &'a [u16],
}

pub struct Linear<'a> {
    pub weight: &'a [u16],
    pub bias: Option<&'a [u16]>,
    pub input: usize,
    pub output: usize,
}

pub struct Layer<'a> {
    pub attention_norm: Norm<'a>,
    pub query: Linear<'a>,
    pub key: Linear<'a>,
    pub value: Linear<'a>,
    pub attention_output: Linear<'a>,
    pub final_norm: Norm<'a>,
    pub fc1: Linear<'a>,
    pub fc2: Linear<'a>,
    pub heads: usize,
    /// `Some(128)` on even pinned layers and `None` on odd pinned layers.
    pub window: Option<usize>,
}

fn norm(input: &[f32], tokens: usize, width: usize, affine: &Norm<'_>) -> Vec<f32> {
    let weight = decode(affine.weight);
    let bias = decode(affine.bias);
    let mut result = vec![0.0; input.len()];
    for row in 0..tokens {
        let values = &input[row * width..(row + 1) * width];
        let mean = values.iter().sum::<f32>() / width as f32;
        let variance = values
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f32>()
            / width as f32;
        let scale = (variance + 1e-5).sqrt().recip();
        for col in 0..width {
            result[row * width + col] =
                bf16(((values[col] - mean) * scale) * weight[col] + bias[col]);
        }
    }
    result
}

fn linear(input: &[f32], tokens: usize, op: &Linear<'_>) -> Vec<f32> {
    let weight = decode(op.weight);
    let bias = op.bias.map(decode);
    let mut result = vec![0.0; tokens * op.output];
    for row in 0..tokens {
        for out in 0..op.output {
            let mut sum = 0.0f32;
            for col in 0..op.input {
                sum = input[row * op.input + col].mul_add(weight[out * op.input + col], sum);
            }
            result[row * op.output + out] = bf16(sum + bias.as_ref().map_or(0.0, |bias| bias[out]));
        }
    }
    result
}

fn rope(values: &mut [f32], tokens: usize, heads: usize, head_size: usize) {
    for row in 0..tokens {
        for head in 0..heads {
            let offset = row * heads * head_size + head * head_size;
            for half in 0..head_size / 2 {
                let angle = row as f32 / 10_000f32.powf((2 * half) as f32 / head_size as f32);
                let (sin, cos) = angle.sin_cos();
                let sin = bf16(sin);
                let cos = bf16(cos);
                let a = values[offset + half];
                let b = values[offset + half + head_size / 2];
                values[offset + half] = bf16(bf16(a * cos) + bf16(-b * sin));
                values[offset + half + head_size / 2] = bf16(bf16(b * cos) + bf16(a * sin));
            }
        }
    }
}

fn attend(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    tokens: usize,
    heads: usize,
    head_size: usize,
    window: Option<usize>,
) -> Vec<f32> {
    let width = heads * head_size;
    let mut output = vec![0.0; tokens * width];
    for row in 0..tokens {
        let first = window.map_or(0, |window| row.saturating_sub(window));
        for head in 0..heads {
            let mut scores = Vec::with_capacity(row - first + 1);
            for past in first..=row {
                let mut score = 0.0f32;
                for col in 0..head_size {
                    score = query[row * width + head * head_size + col]
                        .mul_add(key[past * width + head * head_size + col], score);
                }
                scores.push(score / (head_size as f32).sqrt());
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let normalizer = scores.iter().map(|score| (score - max).exp()).sum::<f32>();
            for col in 0..head_size {
                let mut sum = 0.0f32;
                for (index, past) in (first..=row).enumerate() {
                    let probability = (scores[index] - max).exp() / normalizer;
                    sum = probability.mul_add(value[past * width + head * head_size + col], sum);
                }
                output[row * width + head * head_size + col] = bf16(sum);
            }
        }
    }
    output
}

fn valid_linear(op: &Linear<'_>, input: usize, output: usize, bias: bool) -> bool {
    op.input == input
        && op.output == output
        && op.weight.len() == input * output
        && op.bias.is_some() == bias
        && op.bias.is_none_or(|values| values.len() == output)
        && decode(op.weight).iter().all(|value| value.is_finite())
        && op
            .bias
            .is_none_or(|values| decode(values).iter().all(|value| value.is_finite()))
}

fn valid_norm(op: &Norm<'_>, width: usize) -> bool {
    op.weight.len() == width
        && op.bias.len() == width
        && decode(op.weight).iter().all(|value| value.is_finite())
        && decode(op.bias).iter().all(|value| value.is_finite())
}

/// Execute one source-order pre-norm layer. Input and output carry BF16 values
/// in f32 slots. The window includes the current token and 128 prior positions.
pub fn forward(input: &[f32], tokens: usize, layer: &Layer<'_>) -> Result<Vec<f32>, String> {
    if !(1..=MAX_COMPONENT_TOKENS).contains(&tokens)
        || layer.heads == 0
        || layer.heads > 16
        || layer.query.input == 0
        || layer.query.input > 1_024
        || !layer.query.input.is_multiple_of(layer.heads * 2)
        || !matches!(layer.window, None | Some(128))
    {
        return Err("MiMo codec layer geometry or token count outside component bounds".into());
    }
    let width = layer.query.input;
    let ff = layer.fc1.output;
    if ff == 0
        || ff > 4_096
        || input.len() != tokens * width
        || input
            .iter()
            .any(|value| !value.is_finite() || value.to_bits() & 0xffff != 0)
        || !valid_norm(&layer.attention_norm, width)
        || !valid_norm(&layer.final_norm, width)
        || !valid_linear(&layer.query, width, width, true)
        || !valid_linear(&layer.key, width, width, false)
        || !valid_linear(&layer.value, width, width, true)
        || !valid_linear(&layer.attention_output, width, width, true)
        || !valid_linear(&layer.fc1, width, ff, true)
        || !valid_linear(&layer.fc2, ff, width, true)
    {
        return Err("MiMo codec layer input, BF16 weights, or projection shape changed".into());
    }
    let normalized = norm(input, tokens, width, &layer.attention_norm);
    let mut q = linear(&normalized, tokens, &layer.query);
    let mut k = linear(&normalized, tokens, &layer.key);
    let v = linear(&normalized, tokens, &layer.value);
    let head_size = width / layer.heads;
    rope(&mut q, tokens, layer.heads, head_size);
    rope(&mut k, tokens, layer.heads, head_size);
    let context = attend(&q, &k, &v, tokens, layer.heads, head_size, layer.window);
    let projected = linear(&context, tokens, &layer.attention_output);
    let residual = input
        .iter()
        .zip(projected)
        .map(|(&previous, update)| bf16(previous + update))
        .collect::<Vec<_>>();
    let normalized = norm(&residual, tokens, width, &layer.final_norm);
    let hidden = linear(&normalized, tokens, &layer.fc1)
        .into_iter()
        .map(|value| bf16(gelu_erf(value)))
        .collect::<Vec<_>>();
    let projected = linear(&hidden, tokens, &layer.fc2);
    let result = residual
        .iter()
        .zip(projected)
        .map(|(&previous, update)| bf16(previous + update))
        .collect::<Vec<_>>();
    if result.iter().any(|value| !value.is_finite()) {
        return Err("MiMo codec layer produced a non-finite result".into());
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(value: f32) -> u16 {
        (bf16(value).to_bits() >> 16) as u16
    }

    fn identity(width: usize) -> Vec<u16> {
        (0..width * width)
            .map(|index| {
                bits(if index / width == index % width {
                    1.0
                } else {
                    0.0
                })
            })
            .collect()
    }

    #[test]
    fn causal_layer_keeps_earlier_rows_invariant_under_future_change() {
        let identity = identity(4);
        let zeros = vec![bits(0.0); 4];
        let ones = vec![bits(1.0); 4];
        let mut fc1 = vec![bits(0.0); 8 * 4];
        let fc1_bias = vec![bits(0.0); 8];
        let mut fc2 = vec![bits(0.0); 4 * 8];
        for col in 0..4 {
            fc1[col * 4 + col] = bits(1.0);
            fc2[col * 8 + col] = bits(1.0);
        }
        let no_ff = vec![bits(0.0); 4 * 8];
        let norm = || Norm {
            weight: &ones,
            bias: &zeros,
        };
        let mut layer = Layer {
            attention_norm: norm(),
            query: Linear {
                weight: &identity,
                bias: Some(&zeros),
                input: 4,
                output: 4,
            },
            key: Linear {
                weight: &identity,
                bias: None,
                input: 4,
                output: 4,
            },
            value: Linear {
                weight: &identity,
                bias: Some(&zeros),
                input: 4,
                output: 4,
            },
            attention_output: Linear {
                weight: &identity,
                bias: Some(&zeros),
                input: 4,
                output: 4,
            },
            final_norm: norm(),
            fc1: Linear {
                weight: &fc1,
                bias: Some(&fc1_bias),
                input: 4,
                output: 8,
            },
            fc2: Linear {
                weight: &fc2,
                bias: Some(&zeros),
                input: 8,
                output: 4,
            },
            heads: 2,
            window: Some(128),
        };
        let a = vec![1.0, 0.0, 0.0, -1.0, 0.0, 1.0, -1.0, 0.0];
        let mut b = a.clone();
        b[4..].copy_from_slice(&[2.0, -2.0, 1.0, -1.0]);
        let first = forward(&a, 2, &layer).unwrap();
        let second = forward(&b, 2, &layer).unwrap();
        assert_eq!(&first[..4], &second[..4]);
        assert_ne!(&first[4..], &second[4..]);
        assert!(first.iter().all(|value| value.to_bits() & 0xffff == 0));
        layer.fc2.weight = &no_ff;
        let without_ff = forward(&a, 2, &layer).unwrap();
        assert_ne!(first, without_ff);
    }

    #[test]
    fn local_attention_includes_exactly_128_prior_positions() {
        let tokens = 256;
        let width = 2;
        let mut q = vec![0.0; tokens * width];
        let k = q.clone();
        let mut v = q.clone();
        v[0] = 32.0;
        v[127 * width] = 16.0;
        q[255 * width] = 1.0;
        let full = attend(&q, &k, &v, tokens, 1, 2, None);
        let local = attend(&q, &k, &v, tokens, 1, 2, Some(128));
        assert!(full[255 * width] > local[255 * width]);
        assert!(local[255 * width] > 0.0);
    }

    #[test]
    fn rope_and_bf16_stage_guards() {
        let mut values = vec![1.0, 0.0, 1.0, 0.0];
        rope(&mut values, 2, 1, 2);
        assert_eq!(values[0], 1.0);
        assert!(values[2] < 1.0 && values[3] > 0.0);
        assert!(values.iter().all(|value| value.to_bits() & 0xffff == 0));
        assert_eq!(bf16(1.0 + 1.0 / 256.0), 1.0);
    }

    #[test]
    fn projection_schema_rejects_unpinned_bias_and_nonfinite_weight() {
        let matrix = vec![bits(1.0); 4];
        let bias = vec![bits(0.0); 2];
        let valid = Linear {
            weight: &matrix,
            bias: None,
            input: 2,
            output: 2,
        };
        assert!(valid_linear(&valid, 2, 2, false));
        assert!(!valid_linear(&valid, 2, 2, true));
        let biased = Linear {
            bias: Some(&bias),
            ..valid
        };
        assert!(!valid_linear(&biased, 2, 2, false));
        let nan = [0x7fc0, bits(1.0), bits(1.0), bits(1.0)];
        let corrupted = Linear {
            weight: &nan,
            bias: None,
            input: 2,
            output: 2,
        };
        assert!(!valid_linear(&corrupted, 2, 2, false));
    }
}
