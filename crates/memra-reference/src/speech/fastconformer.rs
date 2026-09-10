//! Native cache-aware FastConformer encoder for the streaming RNNT path.
//!
//! One `stream_step` consumes one chunk of log-mel frames and returns the encoder frames that
//! chunk produced, carrying two caches forward: the per-layer attention history (the normed
//! layer input, which is what the reference caches, not K and V) and the per-layer causal
//! convolution history. Nothing here reads a sample the session does not own: the qualified
//! arm is `[56, 0]`, and a right context would be a different admission.
//!
//! This is a reference operation. It is not a serving route, it is not qualified, and it has
//! no GPU path. The numerics are plain F32; the pinned CPU reference capture is the gate.

use super::encoder::{Linear, Norm, WhisperNumeric, apply_linear, apply_norm};
use memra_gguf::model_packs::nemotron_rnnt::{
    AttentionContext, BoundRnnt, RnntError, RnntGeometry, StreamingStateContract,
};
use memra_gguf::nemo::NemoTensor;

/// LayerNorm epsilon. Torch's default, which is what the checkpoint was trained with.
const NORM_EPSILON: f32 = 1e-5;
/// Half-step residual weight on each feed-forward macaron.
const FF_FACTOR: f32 = 0.5;
/// Base of the sinusoidal position table, from the reference positional encoding.
const POSITION_BASE: f32 = 10000.0;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

struct Conv2d {
    output_channels: usize,
    input_channels: usize,
    kernel: usize,
    stride: usize,
    groups: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl Conv2d {
    /// Causal 2D convolution over `[channels, time, frequency]`, padded `kernel-1` before and
    /// `stride-1` after on both axes, which is what the reference's causal downsampling does.
    fn forward(&self, x: &[f32], time: usize, freq: usize) -> (Vec<f32>, usize, usize) {
        let pad_low = self.kernel - 1;
        let pad_high = self.stride - 1;
        let padded_time = time + pad_low + pad_high;
        let padded_freq = freq + pad_low + pad_high;
        let out_time = (padded_time - self.kernel) / self.stride + 1;
        let out_freq = (padded_freq - self.kernel) / self.stride + 1;
        let per_group = self.input_channels / self.groups;
        let out_per_group = self.output_channels / self.groups;
        let mut y = vec![0.0f32; self.output_channels * out_time * out_freq];
        for out_channel in 0..self.output_channels {
            let group = out_channel / out_per_group;
            let bias = self.bias[out_channel];
            for t in 0..out_time {
                for f in 0..out_freq {
                    let mut sum = bias;
                    for c in 0..per_group {
                        let in_channel = group * per_group + c;
                        for kt in 0..self.kernel {
                            let it = (t * self.stride + kt) as isize - pad_low as isize;
                            if it < 0 || it >= time as isize {
                                continue;
                            }
                            for kf in 0..self.kernel {
                                let iff = (f * self.stride + kf) as isize - pad_low as isize;
                                if iff < 0 || iff >= freq as isize {
                                    continue;
                                }
                                let w = self.weight[((out_channel * per_group + c) * self.kernel
                                    + kt)
                                    * self.kernel
                                    + kf];
                                let v = x[(in_channel * time + it as usize) * freq + iff as usize];
                                sum = w.mul_add(v, sum);
                            }
                        }
                    }
                    y[(out_channel * out_time + t) * out_freq + f] = sum;
                }
            }
        }
        (y, out_time, out_freq)
    }
}

struct Attention {
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    /// `linear_pos(pos_emb)` for the qualified context, folded at load: the position table is
    /// the same for every chunk and every step, so it is not recomputed per frame.
    positions: Vec<f32>,
    position_rows: usize,
    bias_u: Vec<f32>,
    bias_v: Vec<f32>,
}

struct Convolution {
    pointwise1: Linear,
    depthwise: Vec<f32>,
    norm: Norm,
    pointwise2: Linear,
}

struct Block {
    norm_ff1: Norm,
    ff1_in: Linear,
    ff1_out: Linear,
    norm_attention: Norm,
    attention: Attention,
    norm_conv: Norm,
    convolution: Convolution,
    norm_ff2: Norm,
    ff2_in: Linear,
    ff2_out: Linear,
    norm_out: Norm,
}

/// Per-session streaming state. One instance belongs to one audio stream and is never shared.
#[derive(Clone)]
pub struct StreamState {
    /// Per layer, `[cache_frames, width]` of normed layer inputs, oldest first.
    channel: Vec<Vec<f32>>,
    /// Per layer, `[width, kernel - 1]` of convolution history.
    time: Vec<Vec<f32>>,
    /// Cache rows actually written so far. Unwritten rows are masked, not read as zeros.
    valid: usize,
    contract: StreamingStateContract,
}

impl StreamState {
    pub fn new(contract: StreamingStateContract) -> Self {
        let layers = contract.layers as usize;
        let width = contract.width as usize;
        Self {
            channel: vec![vec![0.0; contract.last_channel_frames as usize * width]; layers],
            time: vec![vec![0.0; width * contract.last_time_frames as usize]; layers],
            valid: 0,
            contract,
        }
    }

    pub fn valid_cache_frames(&self) -> usize {
        self.valid
    }

    pub fn contract(&self) -> StreamingStateContract {
        self.contract
    }
}

pub struct FastConformerEncoder {
    geometry: RnntGeometry,
    contract: StreamingStateContract,
    stem: Vec<Conv2d>,
    projection: Linear,
    blocks: Vec<Block>,
}

fn tensor(bound: &BoundRnnt, name: &str, archive: &[u8], storages: &StorageMap) -> Vec<f32> {
    let entry: &NemoTensor = bound
        .tensors
        .get(name)
        .unwrap_or_else(|| panic!("bound checkpoint is missing {name}"));
    let (at, len) = storages[entry.storage_key.as_str()];
    let bytes = &archive[at..at + len];
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

type StorageMap = std::collections::BTreeMap<String, (usize, usize)>;

impl FastConformerEncoder {
    /// Bind an already-verified checkpoint. The caller owns the archive bytes and the bind, so
    /// this cannot silently accept a census that failed its contract.
    pub fn load(
        geometry: RnntGeometry,
        context: AttentionContext,
        bound: &BoundRnnt,
        archive: &[u8],
        storages: &StorageMap,
    ) -> Result<Self, RnntError> {
        let contract = StreamingStateContract::new(geometry, context)?;
        let width = geometry.encoder_width as usize;
        let heads = geometry.encoder_heads as usize;
        let head_width = width / heads;
        let channels = geometry.subsample_channels as usize;
        let get = |name: &str| tensor(bound, name, archive, storages);

        let conv = |index: usize,
                    input: usize,
                    output: usize,
                    kernel: usize,
                    stride: usize,
                    groups: usize| Conv2d {
            output_channels: output,
            input_channels: input,
            kernel,
            stride,
            groups,
            weight: get(&format!("encoder.pre_encode.conv.{index}.weight")),
            bias: get(&format!("encoder.pre_encode.conv.{index}.bias")),
        };
        let stem = vec![
            conv(0, 1, channels, 3, 2, 1),
            conv(2, channels, channels, 3, 2, channels),
            conv(3, channels, channels, 1, 1, 1),
            conv(5, channels, channels, 3, 2, channels),
            conv(6, channels, channels, 1, 1, 1),
        ];
        let projection = Linear {
            input: channels * geometry.subsample_output_bins as usize,
            output: width,
            weight: get("encoder.pre_encode.out.weight"),
            bias: Some(get("encoder.pre_encode.out.bias")),
        };

        // The relative position table the qualified arm needs: one row per relative distance
        // from +cache to -cache around the current frame.
        let span = contract.last_channel_frames as usize + contract.chunk_frames as usize;
        let position_rows = 2 * span - 1;
        let mut table = vec![0.0f32; position_rows * width];
        for row in 0..position_rows {
            let position = (span as f32 - 1.0) - row as f32;
            for pair in 0..width / 2 {
                let scale = (-(POSITION_BASE.ln()) * (2 * pair) as f32 / width as f32).exp();
                table[row * width + 2 * pair] = (position * scale).sin();
                table[row * width + 2 * pair + 1] = (position * scale).cos();
            }
        }

        let linear = |name: &str, input: usize, output: usize| Linear {
            input,
            output,
            weight: get(name),
            bias: None,
        };
        let norm = |name: &str| Norm {
            weight: get(&format!("{name}.weight")),
            bias: get(&format!("{name}.bias")),
        };
        let mut blocks = Vec::with_capacity(geometry.encoder_layers as usize);
        for index in 0..geometry.encoder_layers {
            let p = format!("encoder.layers.{index}");
            let position_projection =
                linear(&format!("{p}.self_attn.linear_pos.weight"), width, width);
            let positions = apply_linear(
                &table,
                position_rows,
                &position_projection,
                WhisperNumeric::F32,
            );
            blocks.push(Block {
                norm_ff1: norm(&format!("{p}.norm_feed_forward1")),
                ff1_in: linear(
                    &format!("{p}.feed_forward1.linear1.weight"),
                    width,
                    geometry.encoder_ffn as usize,
                ),
                ff1_out: linear(
                    &format!("{p}.feed_forward1.linear2.weight"),
                    geometry.encoder_ffn as usize,
                    width,
                ),
                norm_attention: norm(&format!("{p}.norm_self_att")),
                attention: Attention {
                    query: linear(&format!("{p}.self_attn.linear_q.weight"), width, width),
                    key: linear(&format!("{p}.self_attn.linear_k.weight"), width, width),
                    value: linear(&format!("{p}.self_attn.linear_v.weight"), width, width),
                    out: linear(&format!("{p}.self_attn.linear_out.weight"), width, width),
                    positions,
                    position_rows,
                    bias_u: get(&format!("{p}.self_attn.pos_bias_u")),
                    bias_v: get(&format!("{p}.self_attn.pos_bias_v")),
                },
                norm_conv: norm(&format!("{p}.norm_conv")),
                convolution: Convolution {
                    pointwise1: linear(
                        &format!("{p}.conv.pointwise_conv1.weight"),
                        width,
                        2 * width,
                    ),
                    depthwise: get(&format!("{p}.conv.depthwise_conv.weight")),
                    norm: norm(&format!("{p}.conv.batch_norm")),
                    pointwise2: linear(&format!("{p}.conv.pointwise_conv2.weight"), width, width),
                },
                norm_ff2: norm(&format!("{p}.norm_feed_forward2")),
                ff2_in: linear(
                    &format!("{p}.feed_forward2.linear1.weight"),
                    width,
                    geometry.encoder_ffn as usize,
                ),
                ff2_out: linear(
                    &format!("{p}.feed_forward2.linear2.weight"),
                    geometry.encoder_ffn as usize,
                    width,
                ),
                norm_out: norm(&format!("{p}.norm_out")),
            });
        }
        let _ = head_width;
        Ok(Self {
            geometry,
            contract,
            stem,
            projection,
            blocks,
        })
    }

    pub fn contract(&self) -> StreamingStateContract {
        self.contract
    }

    pub fn new_state(&self) -> StreamState {
        StreamState::new(self.contract)
    }

    /// Subsample one chunk of log-mel frames into encoder frames.
    ///
    /// `mel` is `[mel_bins, frames]` in row-major mel-bin order, which is how the reference
    /// feature extractor hands features over.
    pub fn pre_encode(&self, mel: &[f32], frames: usize) -> Result<(Vec<f32>, usize), String> {
        let bins = self.geometry.mel_bins as usize;
        if mel.len() != bins * frames || frames == 0 {
            return Err("FastConformer pre-encode needs [mel_bins, frames]".into());
        }
        // The stem reads [channel, time, frequency]; the features arrive frequency-major.
        let mut x = vec![0.0f32; frames * bins];
        for bin in 0..bins {
            for frame in 0..frames {
                x[frame * bins + bin] = mel[bin * frames + frame];
            }
        }
        let mut time = frames;
        let mut freq = bins;
        for (index, layer) in self.stem.iter().enumerate() {
            let (mut y, out_time, out_freq) = layer.forward(&x, time, freq);
            // The reference activates after the first convolution and after each pointwise.
            if index == 0 || index == 2 || index == 4 {
                for value in y.iter_mut() {
                    *value = value.max(0.0);
                }
            }
            x = y;
            time = out_time;
            freq = out_freq;
        }
        if freq != self.geometry.subsample_output_bins as usize {
            return Err(format!(
                "subsampling left {freq} frequency bins, contract says {}",
                self.geometry.subsample_output_bins
            ));
        }
        // Flatten channel and frequency, channel major, one row per surviving frame.
        let channels = self.geometry.subsample_channels as usize;
        let mut rows = vec![0.0f32; time * channels * freq];
        for t in 0..time {
            for c in 0..channels {
                for f in 0..freq {
                    rows[t * channels * freq + c * freq + f] = x[(c * time + t) * freq + f];
                }
            }
        }
        let projected = apply_linear(&rows, time, &self.projection, WhisperNumeric::F32);
        Ok((projected, time))
    }

    /// One streaming step. `drop_pre_encoded` is the reference's own count of pre-encoded rows
    /// that belong to the previous chunk's overlap and must not be emitted twice.
    pub fn stream_step(
        &self,
        mel: &[f32],
        frames: usize,
        drop_pre_encoded: usize,
        state: &mut StreamState,
    ) -> Result<(Vec<f32>, usize), String> {
        let width = self.geometry.encoder_width as usize;
        let (rows, count) = self.pre_encode(mel, frames)?;
        if drop_pre_encoded > count {
            return Err("chunk produced fewer rows than the drop count".into());
        }
        let emitted = count - drop_pre_encoded;
        if emitted == 0 {
            return Err("chunk emitted no encoder frame".into());
        }
        let mut x = rows[drop_pre_encoded * width..].to_vec();
        let cache_frames = self.contract.last_channel_frames as usize;
        for (index, block) in self.blocks.iter().enumerate() {
            x = self.block_step(block, &x, emitted, index, state)?;
        }
        state.valid = (state.valid + emitted).min(cache_frames);
        let mut out = vec![0.0f32; width * emitted];
        for frame in 0..emitted {
            for channel in 0..width {
                out[channel * emitted + frame] = x[frame * width + channel];
            }
        }
        Ok((out, emitted))
    }

    fn block_step(
        &self,
        block: &Block,
        input: &[f32],
        frames: usize,
        index: usize,
        state: &mut StreamState,
    ) -> Result<Vec<f32>, String> {
        let mut residual = input.to_vec();

        let normed = apply_norm(
            &residual,
            frames,
            &block.norm_ff1,
            WhisperNumeric::F32,
            NORM_EPSILON,
        );
        let mut hidden = apply_linear(&normed, frames, &block.ff1_in, WhisperNumeric::F32);
        for value in hidden.iter_mut() {
            *value = silu(*value);
        }
        let projected = apply_linear(&hidden, frames, &block.ff1_out, WhisperNumeric::F32);
        for (slot, value) in residual.iter_mut().zip(projected) {
            *slot += FF_FACTOR * value;
        }

        let normed = apply_norm(
            &residual,
            frames,
            &block.norm_attention,
            WhisperNumeric::F32,
            NORM_EPSILON,
        );
        let attended = self.attention_step(block, &normed, frames, index, state)?;
        for (slot, value) in residual.iter_mut().zip(attended) {
            *slot += value;
        }

        let normed = apply_norm(
            &residual,
            frames,
            &block.norm_conv,
            WhisperNumeric::F32,
            NORM_EPSILON,
        );
        let convolved = self.convolution_step(block, &normed, frames, index, state);
        for (slot, value) in residual.iter_mut().zip(convolved) {
            *slot += value;
        }

        let normed = apply_norm(
            &residual,
            frames,
            &block.norm_ff2,
            WhisperNumeric::F32,
            NORM_EPSILON,
        );
        let mut hidden = apply_linear(&normed, frames, &block.ff2_in, WhisperNumeric::F32);
        for value in hidden.iter_mut() {
            *value = silu(*value);
        }
        let projected = apply_linear(&hidden, frames, &block.ff2_out, WhisperNumeric::F32);
        for (slot, value) in residual.iter_mut().zip(projected) {
            *slot += FF_FACTOR * value;
        }

        Ok(apply_norm(
            &residual,
            frames,
            &block.norm_out,
            WhisperNumeric::F32,
            NORM_EPSILON,
        ))
    }

    /// Relative-position attention over the cached history plus this chunk.
    fn attention_step(
        &self,
        block: &Block,
        normed: &[f32],
        frames: usize,
        index: usize,
        state: &mut StreamState,
    ) -> Result<Vec<f32>, String> {
        let width = self.geometry.encoder_width as usize;
        let heads = self.geometry.encoder_heads as usize;
        let head_width = width / heads;
        let cache_frames = self.contract.last_channel_frames as usize;
        let attention = &block.attention;
        if frames != self.contract.chunk_frames as usize || frames != 1 {
            // The relative-position shift collapses to a direct distance lookup only for one
            // query row. The qualified arm emits exactly one frame per chunk; a wider chunk
            // needs the general shift and its own gate, so it is refused rather than guessed.
            return Err(format!(
                "cache-aware attention is gated for one frame per chunk, got {frames}"
            ));
        }

        // Keys and values see the cache first, then this chunk, exactly as the reference
        // concatenates them before projecting.
        let mut keyed = Vec::with_capacity((cache_frames + frames) * width);
        keyed.extend_from_slice(&state.channel[index]);
        keyed.extend_from_slice(normed);
        let total = cache_frames + frames;

        let query = apply_linear(normed, frames, &attention.query, WhisperNumeric::F32);
        let key = apply_linear(&keyed, total, &attention.key, WhisperNumeric::F32);
        let value = apply_linear(&keyed, total, &attention.value, WhisperNumeric::F32);

        // Rows the cache has not written yet carry no audio and are masked, not read as zeros.
        let first_valid = cache_frames - state.valid.min(cache_frames);
        let scale = 1.0 / (head_width as f32).sqrt();
        let mut context = vec![0.0f32; frames * width];
        let mut scores = vec![0.0f32; total];
        for head in 0..heads {
            for q in 0..frames {
                let query_row =
                    &query[q * width + head * head_width..q * width + (head + 1) * head_width];
                let absolute_query = cache_frames + q;
                for k in 0..total {
                    if k < first_valid || k > absolute_query {
                        scores[k] = f32::NEG_INFINITY;
                        continue;
                    }
                    let key_row =
                        &key[k * width + head * head_width..k * width + (head + 1) * head_width];
                    let mut ac = 0.0f32;
                    let mut bd = 0.0f32;
                    // Relative distance, mapped onto the position table's centre row.
                    let distance = absolute_query as isize - k as isize;
                    let row = (self.contract.last_channel_frames as isize
                        + self.contract.chunk_frames as isize
                        - 1
                        - distance) as usize;
                    if row >= attention.position_rows {
                        return Err("relative position falls outside the pinned table".into());
                    }
                    let position_row = &attention.positions
                        [row * width + head * head_width..row * width + (head + 1) * head_width];
                    for d in 0..head_width {
                        let u = attention.bias_u[head * head_width + d];
                        let v = attention.bias_v[head * head_width + d];
                        ac = (query_row[d] + u).mul_add(key_row[d], ac);
                        bd = (query_row[d] + v).mul_add(position_row[d], bd);
                    }
                    scores[k] = (ac + bd) * scale;
                }
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                if !maximum.is_finite() {
                    return Err("attention row has no admissible key".into());
                }
                let mut sum = 0.0f32;
                for score in scores.iter_mut() {
                    *score = if score.is_finite() {
                        (*score - maximum).exp()
                    } else {
                        0.0
                    };
                    sum += *score;
                }
                let inverse = sum.recip();
                for k in 0..total {
                    let weight = scores[k] * inverse;
                    if weight == 0.0 {
                        continue;
                    }
                    let value_row =
                        &value[k * width + head * head_width..k * width + (head + 1) * head_width];
                    for d in 0..head_width {
                        context[q * width + head * head_width + d] = weight
                            .mul_add(value_row[d], context[q * width + head * head_width + d]);
                    }
                }
            }
        }
        let out = apply_linear(&context, frames, &attention.out, WhisperNumeric::F32);

        // The cache keeps the normed layer input, oldest row dropped, this chunk appended.
        let cache = &mut state.channel[index];
        if frames >= cache_frames {
            cache.copy_from_slice(&normed[(frames - cache_frames) * width..]);
        } else {
            cache.copy_within(frames * width.., 0);
            cache[(cache_frames - frames) * width..].copy_from_slice(normed);
        }
        Ok(out)
    }

    /// Causal depthwise convolution module with its own history.
    fn convolution_step(
        &self,
        block: &Block,
        normed: &[f32],
        frames: usize,
        index: usize,
        state: &mut StreamState,
    ) -> Vec<f32> {
        let width = self.geometry.encoder_width as usize;
        let kernel = self.geometry.conv_kernel as usize;
        let history = kernel - 1;
        let convolution = &block.convolution;

        let gated_input =
            apply_linear(normed, frames, &convolution.pointwise1, WhisperNumeric::F32);
        let mut gated = vec![0.0f32; frames * width];
        for frame in 0..frames {
            for channel in 0..width {
                let a = gated_input[frame * 2 * width + channel];
                let b = gated_input[frame * 2 * width + width + channel];
                gated[frame * width + channel] = a * sigmoid(b);
            }
        }

        // History first, then this chunk, channel major, exactly as the cache is stored.
        let mut series = vec![0.0f32; width * (history + frames)];
        for channel in 0..width {
            for t in 0..history {
                series[channel * (history + frames) + t] = state.time[index][channel * history + t];
            }
            for t in 0..frames {
                series[channel * (history + frames) + history + t] = gated[t * width + channel];
            }
        }
        let mut convolved = vec![0.0f32; frames * width];
        for channel in 0..width {
            for t in 0..frames {
                let mut sum = 0.0f32;
                for k in 0..kernel {
                    let w = convolution.depthwise[channel * kernel + k];
                    sum = w.mul_add(series[channel * (history + frames) + t + k], sum);
                }
                convolved[t * width + channel] = sum;
            }
        }
        // Carry the last `history` frames of the padded series forward.
        for channel in 0..width {
            for t in 0..history {
                state.time[index][channel * history + t] =
                    series[channel * (history + frames) + frames + t];
            }
        }

        let normalized = apply_norm(
            &convolved,
            frames,
            &convolution.norm,
            WhisperNumeric::F32,
            NORM_EPSILON,
        );
        let mut activated = normalized;
        for value in activated.iter_mut() {
            *value = silu(*value);
        }
        apply_linear(
            &activated,
            frames,
            &convolution.pointwise2,
            WhisperNumeric::F32,
        )
    }
}

#[cfg(test)]
mod tests;
