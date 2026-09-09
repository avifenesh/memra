//! Native Whisper convolution stem and bidirectional encoder. No external model executor.
//! FP16 mode uses binary16 weights/activations and FP32 accumulation. The CPU reference
//! carries rounded binary16 values in f32 slots; it is not a CUDA performance receipt.

use super::matrix;
use crate::ReferenceTensor;
use memra_gguf::dequant::fp16_to_f32;
use memra_gguf::model_plan::speech::{AudioPositionKind, SpeechActivation, WhisperPlan};
use memra_gguf::model_plan::{NormKind, WeightTransform};
use memra_gguf::nvfp4_repack::f32_to_f16_bits;
use memra_gguf::safetensors::StModel;
use memra_gguf::tensor_contract::{CheckpointDialect, FloatType, StorageLayout, TensorCensusEntry};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhisperNumeric {
    F32,
    F16,
}

impl WhisperNumeric {
    #[inline]
    pub fn round(self, x: f32) -> f32 {
        if self == Self::F32 {
            return x;
        }
        let h = f32_to_f16_bits(x);
        let exponent = (h >> 10) & 31;
        if (1..31).contains(&exponent) {
            f32::from_bits(
                (u32::from(h & 0x8000) << 16)
                    | ((u32::from(exponent) + 112) << 23)
                    | (u32::from(h & 1023) << 13),
            )
        } else {
            fp16_to_f32(h)
        }
    }
}

struct Linear {
    input: usize,
    output: usize,
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
}

struct Norm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

struct EncoderBlock {
    attention_norm: Norm,
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    mlp_norm: Norm,
    fc1: Linear,
    fc2: Linear,
}

pub struct WhisperEncoder {
    plan: WhisperPlan,
    numeric: WhisperNumeric,
    conv1: Linear,
    conv2: Linear,
    positions: Vec<f32>,
    blocks: Vec<EncoderBlock>,
    output_norm: Norm,
}

fn tensor(
    source: &StModel,
    name: &str,
    shape: &[usize],
    numeric: WhisperNumeric,
) -> Result<Vec<f32>, String> {
    let (info, raw) = source
        .raw(name)
        .ok_or_else(|| format!("missing Whisper encoder tensor {name}"))?;
    if info.dtype != "F32" || info.shape != shape.iter().map(|&x| x as u64).collect::<Vec<_>>() {
        return Err(format!(
            "Whisper tensor {name}: expected F32 {shape:?}, got {} {:?}",
            info.dtype, info.shape
        ));
    }
    let mut values = Vec::with_capacity(raw.len() / 4);
    for b in raw.chunks_exact(4) {
        let x = numeric.round(f32::from_le_bytes(b.try_into().unwrap()));
        if !x.is_finite() {
            return Err(format!(
                "Whisper tensor {name}: non-finite value after {numeric:?} conversion"
            ));
        }
        values.push(x);
    }
    Ok(values)
}

fn load_linear(
    source: &StModel,
    stem: &str,
    input: usize,
    output: usize,
    bias: bool,
    numeric: WhisperNumeric,
) -> Result<Linear, String> {
    Ok(Linear {
        input,
        output,
        weight: tensor(source, &format!("{stem}.weight"), &[output, input], numeric)?,
        bias: if bias {
            Some(tensor(source, &format!("{stem}.bias"), &[output], numeric)?)
        } else {
            None
        },
    })
}

fn load_norm(
    source: &StModel,
    stem: &str,
    hidden: usize,
    numeric: WhisperNumeric,
) -> Result<Norm, String> {
    Ok(Norm {
        weight: tensor(source, &format!("{stem}.weight"), &[hidden], numeric)?,
        bias: tensor(source, &format!("{stem}.bias"), &[hidden], numeric)?,
    })
}

fn validate(plan: &WhisperPlan) -> Result<(), String> {
    let h = plan.hidden_size;
    if h == 0
        || h > 1280
        || plan.encoder_layers == 0
        || plan.encoder_layers > 32
        || plan.encoder_heads == 0
        || plan.encoder_heads > 20
        || !h.is_multiple_of(plan.encoder_heads)
        || plan.encoder_ffn == 0
        || plan.encoder_ffn > 5120
        || plan.source_positions == 0
        || plan.source_positions > 1500
        || plan.frontend.max_frames != 2 * plan.source_positions
        || ![80, 128].contains(&plan.frontend.mel_bins)
        || plan.norm.kind != NormKind::LayerNorm
        || plan.norm.epsilon != 1e-5
        || plan.norm.weight_transform != WeightTransform::Identity
        || plan.activation != SpeechActivation::GeluErf
        || plan.encoder_position != AudioPositionKind::CheckpointSinusoidal
        || !plan.attention.encoder_bidirectional
        || !plan.attention.pre_norm_serial_residual
        || !plan.attention.query_bias
        || plan.attention.key_bias
        || !plan.attention.value_bias
        || !plan.attention.output_bias
        || plan.attention.qk_scale != 1.0 / ((h / plan.encoder_heads) as f32).sqrt()
    {
        return Err("unsupported Whisper encoder program or geometry".into());
    }
    for (i, conv) in plan.conv_stem.iter().enumerate() {
        let channels = if i == 0 { plan.frontend.mel_bins } else { h };
        if conv.input_channels != channels
            || conv.output_channels != h
            || conv.kernel != 3
            || conv.padding != 1
            || conv.stride != (i + 1) as u32
            || !conv.bias
        {
            return Err("unsupported Whisper convolution stem".into());
        }
    }
    Ok(())
}

impl WhisperEncoder {
    /// Bind exactly the encoder subgraph. Full checkpoints and explicitly encoder-only
    /// synthetic fixtures share the same contract; decoder tensors never count as encoder proof.
    pub fn load(
        plan: &WhisperPlan,
        source: &StModel,
        numeric: WhisperNumeric,
    ) -> Result<Self, String> {
        validate(plan)?;
        let mut contract = memra_gguf::model_packs::whisper::tensor_contract(
            plan,
            CheckpointDialect::HfSafetensors,
        )
        .map_err(|e| e.to_string())?;
        contract
            .requirements
            .retain(|r| r.names.iter().all(|n| n.starts_with("model.encoder.")));
        let census = source
            .names()
            .filter(|n| n.starts_with("model.encoder."))
            .map(|name| {
                let info = source.info(name).unwrap();
                if info.dtype != "F32" {
                    return Err(format!("encoder source tensor {name} is not F32"));
                }
                Ok(TensorCensusEntry {
                    name: name.clone(),
                    shape: info.shape.clone(),
                    storage: StorageLayout::Float(FloatType::F32),
                    physical_bytes: (info.data_offsets[1] - info.data_offsets[0]) as u64,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        contract.bind(&census).map_err(|e| e.to_string())?;
        let h = plan.hidden_size as usize;
        let ffn = plan.encoder_ffn as usize;
        let load_conv = |i: usize, channels: usize| -> Result<Linear, String> {
            let stem = format!("model.encoder.conv{i}");
            Ok(Linear {
                input: channels * 3,
                output: h,
                weight: tensor(
                    source,
                    &format!("{stem}.weight"),
                    &[h, channels, 3],
                    numeric,
                )?,
                bias: Some(tensor(source, &format!("{stem}.bias"), &[h], numeric)?),
            })
        };
        let conv1 = load_conv(1, plan.frontend.mel_bins as usize)?;
        let conv2 = load_conv(2, h)?;
        let mut blocks = Vec::with_capacity(plan.encoder_layers as usize);
        for i in 0..plan.encoder_layers {
            let p = format!("model.encoder.layers.{i}");
            blocks.push(EncoderBlock {
                attention_norm: load_norm(
                    source,
                    &format!("{p}.self_attn_layer_norm"),
                    h,
                    numeric,
                )?,
                query: load_linear(
                    source,
                    &format!("{p}.self_attn.q_proj"),
                    h,
                    h,
                    true,
                    numeric,
                )?,
                key: load_linear(
                    source,
                    &format!("{p}.self_attn.k_proj"),
                    h,
                    h,
                    false,
                    numeric,
                )?,
                value: load_linear(
                    source,
                    &format!("{p}.self_attn.v_proj"),
                    h,
                    h,
                    true,
                    numeric,
                )?,
                out: load_linear(
                    source,
                    &format!("{p}.self_attn.out_proj"),
                    h,
                    h,
                    true,
                    numeric,
                )?,
                mlp_norm: load_norm(source, &format!("{p}.final_layer_norm"), h, numeric)?,
                fc1: load_linear(source, &format!("{p}.fc1"), h, ffn, true, numeric)?,
                fc2: load_linear(source, &format!("{p}.fc2"), ffn, h, true, numeric)?,
            });
        }
        Ok(Self {
            plan: plan.clone(),
            numeric,
            conv1,
            conv2,
            blocks,
            positions: tensor(
                source,
                "model.encoder.embed_positions.weight",
                &[plan.source_positions as usize, h],
                numeric,
            )?,
            output_norm: load_norm(source, "model.encoder.layer_norm", h, numeric)?,
        })
    }

    /// Gate probe for the exact first encoder norm, on captured pre-norm rows.
    pub fn probe_first_norm(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        let h = self.plan.hidden_size as usize;
        if input.is_empty()
            || !input.len().is_multiple_of(h)
            || input.iter().any(|x| !x.is_finite())
        {
            return Err("invalid norm probe input".into());
        }
        Ok(self.norm(input, input.len() / h, &self.blocks[0].attention_norm))
    }

    pub fn encode(&self, mel: &ReferenceTensor) -> Result<ReferenceTensor, String> {
        self.encode_with_trace(mel, |_, _| Ok(()))
    }

    /// Trace destinations are borrowed, making full-stage oracle captures optional without
    /// retaining every layer in memory. A trace write failure fails the capture.
    pub fn encode_with_trace(
        &self,
        mel: &ReferenceTensor,
        mut trace: impl FnMut(&str, &ReferenceTensor) -> Result<(), String>,
    ) -> Result<ReferenceTensor, String> {
        let frames = self.plan.frontend.max_frames as usize;
        let channels = self.plan.frontend.mel_bins as usize;
        let h = self.plan.hidden_size as usize;
        if mel.shape != [channels, frames]
            || mel.data.len() != channels * frames
            || mel.data.iter().any(|x| !x.is_finite())
        {
            return Err(format!(
                "Whisper encoder requires finite [{channels},{frames}] log-mel"
            ));
        }
        let mut x = vec![0.0f32; frames * channels];
        for row in 0..frames {
            for c in 0..channels {
                x[row * channels + c] = self.numeric.round(mel.data[c * frames + row]);
            }
        }
        let conv1 = self.conv(&x, frames, channels, &self.conv1, 1);
        trace(
            "conv1",
            &ReferenceTensor {
                shape: vec![frames, h],
                data: conv1.clone(),
                ints: None,
            },
        )?;
        let mut x = self.conv(&conv1, frames, h, &self.conv2, 2);
        let rows = frames / 2;
        trace(
            "conv2",
            &ReferenceTensor {
                shape: vec![rows, h],
                data: x.clone(),
                ints: None,
            },
        )?;
        for (value, &position) in x.iter_mut().zip(&self.positions) {
            *value = self.numeric.round(*value + position);
        }
        for (i, block) in self.blocks.iter().enumerate() {
            let normalized = self.norm(&x, rows, &block.attention_norm);
            let attention = self.attention(&normalized, rows, block);
            let attention = self.linear(&attention, rows, &block.out);
            for (value, residual) in x.iter_mut().zip(attention) {
                *value = self.numeric.round(*value + residual);
            }
            let normalized = self.norm(&x, rows, &block.mlp_norm);
            let mut ff = self.linear(&normalized, rows, &block.fc1);
            self.gelu(&mut ff);
            let ff = self.linear(&ff, rows, &block.fc2);
            for (value, residual) in x.iter_mut().zip(ff) {
                *value = self.numeric.round(*value + residual);
                if self.numeric == WhisperNumeric::F16 {
                    *value = self.numeric.round(value.clamp(-64504.0, 64504.0));
                }
            }
            if x.iter().any(|x| !x.is_finite()) {
                return Err(format!("non-finite Whisper encoder layer {i}"));
            }
            trace(
                &format!("layer-{i:02}"),
                &ReferenceTensor {
                    shape: vec![rows, h],
                    data: x.clone(),
                    ints: None,
                },
            )?;
        }
        let output = self.norm(&x, rows, &self.output_norm);
        if output.iter().any(|x| !x.is_finite()) {
            return Err("non-finite Whisper encoder output".into());
        }
        let output = ReferenceTensor {
            shape: vec![rows, h],
            data: output,
            ints: None,
        };
        trace("encoder", &output)?;
        Ok(output)
    }

    fn linear(&self, x: &[f32], rows: usize, layer: &Linear) -> Vec<f32> {
        let mut y = matrix::linear(x, &layer.weight, rows, layer.input, layer.output);
        for (i, value) in y.iter_mut().enumerate() {
            let bias = layer.bias.as_ref().map_or(0.0, |b| b[i % layer.output]);
            *value = self.numeric.round(*value + bias);
        }
        y
    }

    fn conv(
        &self,
        x: &[f32],
        rows: usize,
        channels: usize,
        layer: &Linear,
        stride: usize,
    ) -> Vec<f32> {
        let output_rows = rows.div_ceil(stride);
        let mut patches = vec![0.0f32; output_rows * channels * 3];
        for row in 0..output_rows {
            for c in 0..channels {
                for tap in 0..3 {
                    let source = (row * stride + tap) as isize - 1;
                    if source >= 0 && (source as usize) < rows {
                        patches[(row * channels + c) * 3 + tap] = x[source as usize * channels + c];
                    }
                }
            }
        }
        let mut output = self.linear(&patches, output_rows, layer);
        self.gelu(&mut output);
        output
    }

    fn norm(&self, x: &[f32], rows: usize, norm: &Norm) -> Vec<f32> {
        let h = norm.weight.len();
        let mut output = vec![0.0; x.len()];
        for row in 0..rows {
            let values = &x[row * h..(row + 1) * h];
            let (mean, variance) = row_moments(values, self.numeric == WhisperNumeric::F16);
            let inv = (variance + self.plan.norm.epsilon).sqrt().recip();
            for c in 0..h {
                let normalized = if self.numeric == WhisperNumeric::F16 {
                    // The pinned HF reduced-precision LayerNorm forms scale and bias in
                    // FP32 before applying the affine. Do not reassociate this as x-mean.
                    values[c] * inv + (-inv * mean)
                } else {
                    (values[c] - mean) * inv
                };
                output[row * h + c] = self
                    .numeric
                    .round(normalized * norm.weight[c] + norm.bias[c]);
            }
        }
        output
    }

    fn gelu(&self, x: &mut [f32]) {
        for value in x {
            *value = self.numeric.round(gelu_erf(*value));
        }
    }

    fn attention(&self, x: &[f32], rows: usize, block: &EncoderBlock) -> Vec<f32> {
        let h = self.plan.hidden_size as usize;
        let heads = self.plan.encoder_heads as usize;
        let dim = h / heads;
        let mut query = self.linear(x, rows, &block.query);
        for q in &mut query {
            *q = self.numeric.round(*q * self.plan.attention.qk_scale);
        }
        let key = self.linear(x, rows, &block.key);
        let value = self.linear(x, rows, &block.value);
        let mut output = vec![0.0; rows * h];
        let mut q = vec![0.0; rows * dim];
        let mut k = vec![0.0; rows * dim];
        let mut vt = vec![0.0; dim * rows];
        for head in 0..heads {
            for row in 0..rows {
                for d in 0..dim {
                    q[row * dim + d] = query[row * h + head * dim + d];
                    k[row * dim + d] = key[row * h + head * dim + d];
                    vt[d * rows + row] = value[row * h + head * dim + d];
                }
            }
            let mut scores = matrix::linear(&q, &k, rows, dim, rows);
            for row in scores.chunks_exact_mut(rows) {
                for score in row.iter_mut() {
                    *score = self.numeric.round(*score);
                }
                let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut partials = [0.0f32; 8];
                let vector_end = if self.numeric == WhisperNumeric::F16 {
                    row.len() / 8 * 8
                } else {
                    row.len()
                };
                for (i, score) in row.iter_mut().enumerate() {
                    *score = (*score - maximum).exp();
                    if i < vector_end {
                        partials[i % 8] += *score;
                    }
                }
                if self.numeric == WhisperNumeric::F16 {
                    // Pinned HF AVX2 half softmax: cross-half, pair, then adjacent reduction;
                    // scalar tail follows that reduction. Normalization multiplies one FP32
                    // reciprocal, rather than separately rounding a division per probability.
                    let mut sum = ((partials[0] + partials[4]) + (partials[2] + partials[6]))
                        + ((partials[1] + partials[5]) + (partials[3] + partials[7]));
                    for &value in &row[vector_end..] {
                        sum += value;
                    }
                    let inverse = sum.recip();
                    for score in row {
                        *score = self.numeric.round(*score * inverse);
                    }
                } else {
                    let sum = ((partials[0] + partials[1]) + (partials[2] + partials[3]))
                        + ((partials[4] + partials[5]) + (partials[6] + partials[7]));
                    for score in row {
                        *score /= sum;
                    }
                }
            }
            let context = matrix::linear(&scores, &vt, rows, rows, dim);
            for row in 0..rows {
                for d in 0..dim {
                    output[row * h + head * dim + d] = self.numeric.round(context[row * dim + d]);
                }
            }
        }
        output
    }
}

/// Erf GELU using the published Abramowitz-Stegun 7.1.26 approximation in FP32.
/// This is the pinned HF CPU vector program, including fused Horner evaluation.
/// No upstream kernel is linked or vendored. It is distinct from tanh GELU.
pub fn gelu_erf(x: f32) -> f32 {
    let z = x * std::f32::consts::FRAC_1_SQRT_2;
    let t = 1.0 / 0.327_591_1f32.mul_add(z.abs(), 1.0);
    let p = 1.061_405_4f32.mul_add(t, -1.453_152_1);
    let p = p.mul_add(t, 1.421_413_8);
    let p = p.mul_add(t, -0.284_496_72);
    let p = p.mul_add(t, 0.254_829_6);
    let tail = -(-z * z).exp() * t;
    let erf = tail.mul_add(p, 1.0).copysign(z);
    (x * 0.5) * (1.0 + erf)
}

/// Blockwise Welford moments. Short strided accumulators and weighted merges keep
/// cancellation in long LayerNorm rows from entering the residual stream.
fn row_moments(values: &[f32], reduced: bool) -> (f32, f32) {
    #[derive(Clone, Copy, Default)]
    struct Moments {
        count: usize,
        mean: f32,
        m2: f32,
    }
    impl Moments {
        fn push(&mut self, x: f32) {
            self.count += 1;
            let delta = x - self.mean;
            self.mean = (1.0 / self.count as f32).mul_add(delta, self.mean);
            self.m2 = delta.mul_add(x - self.mean, self.m2);
        }
        fn merge(&mut self, other: Self) {
            if other.count == 0 {
                return;
            }
            let total = self.count + other.count;
            let fraction = other.count as f32 / total as f32;
            let delta = other.mean - self.mean;
            self.mean += fraction * delta;
            self.m2 = (delta * self.count as f32).mul_add(fraction * delta, self.m2 + other.m2);
            self.count = total;
        }
    }
    const LANES: usize = 8;
    let vector_width = if reduced { 2 * LANES } else { LANES };
    let vectors = values.len() / vector_width;
    let mut stack = [[Moments::default(); LANES]; 8];
    let mut groups = 0usize;
    for start in (0..vectors).step_by(16) {
        let mut low = [Moments::default(); LANES];
        let mut high = [Moments::default(); LANES];
        for v in start..(start + 16).min(vectors) {
            for lane in 0..LANES {
                low[lane].push(values[v * vector_width + lane]);
                if reduced {
                    high[lane].push(values[v * vector_width + LANES + lane]);
                }
            }
        }
        for lane in 0..LANES {
            stack[0][lane].merge(low[lane]);
            if reduced {
                stack[0][lane].merge(high[lane]);
            }
        }
        groups += 1;
        let mut carries = groups;
        let mut level = 1;
        while carries.is_multiple_of(2) && level < stack.len() {
            for lane in 0..LANES {
                let lower = stack[level - 1][lane];
                stack[level][lane].merge(lower);
                stack[level - 1][lane] = Moments::default();
            }
            carries /= 2;
            level += 1;
        }
    }
    for level in 1..stack.len() {
        for lane in 0..LANES {
            let other = stack[level][lane];
            stack[0][lane].merge(other);
        }
    }
    let mut merged = Moments::default();
    for &x in &values[vectors * vector_width..] {
        merged.push(x);
    }
    for lane in 0..LANES {
        merged.merge(stack[0][lane]);
    }
    (merged.mean, merged.m2 / values.len() as f32)
}

#[cfg(test)]
mod tests;
