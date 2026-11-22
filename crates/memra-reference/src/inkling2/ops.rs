//! The Inkling 2 operations, one function each, in f64.
//!
//! Every function here is the math of one operation of the plan types in
//! `memra_gguf::model_plan::inkling2`, unfused, on one row at a time. Weights are read as the f32
//! values the checkpoint stores (BF16 and F32 tensors are exact in f32) and widened; the routed
//! expert banks are read from their stored bytes and dequantized exactly. Sums run in ascending
//! index order. Nothing here is a model of a kernel: tuned paths are compared with these.
//!
//! Each function is pinned to the module of the same role in the checkpoint's float-activation
//! implementation, run in f64 (`fixtures/inkling2-module-oracles.bin`, written by
//! `research/modelplan-onboarding-inkling2mini-20261121/reference/module_oracles.py`).

use crate::inkling2_actquant::{e2m1_decode, e4m3_decode};

/// A plan scalar as the decimal its config wrote.
///
/// Plan scalars are f32. The shortest decimal that names the same f32 is the config's own
/// literal for every value this family declares (1e-6, 0.1, 4, 25, 30, 8) and within 1e-9 of it
/// for the Q/K clip threshold, whose config literal has more digits than f32 holds.
pub fn decimal(value: f32) -> f64 {
    value
        .to_string()
        .parse()
        .expect("an f32 prints as a decimal f64 parses")
}

/// `[rows, input]` weight times one input row.
pub fn linear(weight: &[f32], x: &[f64]) -> Vec<f64> {
    debug_assert!(weight.len().is_multiple_of(x.len()));
    weight
        .chunks_exact(x.len())
        .map(|row| dot_f32(row, x))
        .collect()
}

fn dot_f32(weight: &[f32], x: &[f64]) -> f64 {
    let mut sum = 0.0;
    for (w, v) in weight.iter().zip(x) {
        sum += f64::from(*w) * v;
    }
    sum
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    for (x, y) in a.iter().zip(b) {
        sum += x * y;
    }
    sum
}

/// RMS norm: `x * rsqrt(mean(x^2) + epsilon)`, times the learned weight when there is one.
pub fn rms_norm(x: &[f64], weight: Option<&[f32]>, epsilon: f64) -> Vec<f64> {
    let mean_square = dot(x, x) / x.len() as f64;
    let factor = 1.0 / (mean_square + epsilon).sqrt();
    match weight {
        Some(weight) => x
            .iter()
            .zip(weight)
            .map(|(v, w)| f64::from(*w) * (v * factor))
            .collect(),
        None => x.iter().map(|v| v * factor).collect(),
    }
}

/// L2 clip of one head vector: `x * min(1, threshold / (|x| + epsilon))`.
pub fn l2_clip(x: &mut [f64], threshold: f64, epsilon: f64) {
    let norm = dot(x, x).sqrt();
    let factor = (threshold / (norm + epsilon)).min(1.0);
    for value in x {
        *value *= factor;
    }
}

/// Clip every head of `rows` (`[heads, width]`): the optional learned per-channel scale, the L2
/// clip, then the log position scale `tau` (1 where the layer has none).
pub fn clip_heads(
    rows: &mut [f64],
    width: usize,
    threshold: f64,
    epsilon: f64,
    scale: Option<&[f32]>,
    tau: f64,
) {
    for (head, row) in rows.chunks_exact_mut(width).enumerate() {
        if let Some(scale) = scale {
            for (value, factor) in row.iter_mut().zip(&scale[head * width..]) {
                *value *= f64::from(*factor);
            }
        }
        l2_clip(row, threshold, epsilon);
        for value in row.iter_mut() {
            *value *= tau;
        }
    }
}

/// The bias of one query against every distance: `bias[h, d] = (r_h @ table[rank, extent])[d]`
/// for the query's relative vectors `relative` (`[heads, rank]`, after their clip).
pub fn relative_bias(relative: &[f64], table: &[f32], rank: usize, extent: usize) -> Vec<f64> {
    let heads = relative.len() / rank;
    let mut bias = vec![0.0; heads * extent];
    for head in 0..heads {
        for distance in 0..extent {
            let mut sum = 0.0;
            for r in 0..rank {
                sum += relative[head * rank + r] * f64::from(table[r * extent + distance]);
            }
            bias[head * extent + distance] = sum;
        }
    }
    bias
}

/// The four message queries of core layer `index`, in `[q, k, v, mlp]` order, from the
/// checkpoint's two tensors: `query_mlp` holds q at row `2i` and mlp at row `2i + 1`, and
/// `key_value` holds k at row `2i` and v at row `2i + 1`.
pub fn message_queries(
    query_mlp: &[f32],
    key_value: &[f32],
    index: usize,
    hidden: usize,
) -> Vec<f32> {
    let row = |tensor: &[f32], row: usize| tensor[row * hidden..(row + 1) * hidden].to_vec();
    let mut queries = row(query_mlp, 2 * index);
    queries.extend(row(key_value, 2 * index));
    queries.extend(row(key_value, 2 * index + 1));
    queries.extend(row(query_mlp, 2 * index + 1));
    queries
}

/// The log position scale: `1 + alpha * ln(max(1, (position + 1) / floor))`.
pub fn log_position_scale(position: u64, floor: u32, alpha: f64) -> f64 {
    1.0 + alpha * ((position + 1) as f64 / f64::from(floor)).max(1.0).ln()
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// `cap * tanh(x / cap)`.
pub fn softcap(x: f64, cap: f64) -> f64 {
    (x / cap).tanh() * cap
}

/// Softcapped SwiGLU: the sigmoid reads the raw gate; the caps bound the linear gate factor and
/// the up value.
pub fn softcapped_swiglu(gate: f64, up: f64, gate_cap: f64, up_cap: f64) -> f64 {
    softcap(gate, gate_cap) * sigmoid(gate) * softcap(up, up_cap)
}

/// The last `kernel - 1` inputs of one short convolution, oldest first. Inputs before position
/// 0 are zero.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ConvHistory {
    rows: Vec<Vec<f64>>,
}

impl ConvHistory {
    /// The input `lag` steps back (`lag >= 1`); `None` before position 0.
    fn back(&self, lag: usize) -> Option<&[f64]> {
        self.rows
            .len()
            .checked_sub(lag)
            .map(|index| self.rows[index].as_slice())
    }

    fn push(&mut self, row: &[f64], kernel: usize) {
        self.rows.push(row.to_vec());
        let keep = kernel - 1;
        if self.rows.len() > keep {
            let drop = self.rows.len() - keep;
            self.rows.drain(..drop);
        }
    }

    /// The kept rows, oldest first, flattened.
    pub fn flat(&self) -> Vec<f64> {
        self.rows.iter().flatten().copied().collect()
    }
}

/// Static short convolution of one input row: `(x + sum_j taps[c, j] * x[t - (kernel - 1) + j]) / sqrt(kernel + 1)`.
///
/// `taps` is the checkpoint's `[channels, 1, kernel]` tensor: a causal cross-correlation, so
/// its LAST tap weights the current token.
pub fn short_conv_static(
    x: &[f64],
    history: &mut ConvHistory,
    taps: &[f32],
    kernel: usize,
) -> Vec<f64> {
    let divisor = ((kernel + 1) as f64).sqrt();
    let mut out = vec![0.0; x.len()];
    for (channel, value) in out.iter_mut().enumerate() {
        let mut sum = 0.0;
        // oldest first, as the cross-correlation runs
        for tap in 0..kernel {
            let lag = kernel - 1 - tap;
            let input = if lag == 0 {
                x[channel]
            } else {
                history.back(lag).map_or(0.0, |row| row[channel])
            };
            sum += f64::from(taps[channel * kernel + tap]) * input;
        }
        *value = (sum + x[channel]) / divisor;
    }
    history.push(x, kernel);
    out
}

/// Dynamic short convolution of one input row:
/// `(x + sum_lag (static_taps[lag, c] + dynamic[group(c), lag]) * x[t - lag]) / sqrt(2 * kernel + 1)`.
///
/// Both tap sets are indexed by lag with the current token FIRST, the reverse of
/// [`short_conv_static`]. `static_taps` is the checkpoint's `[kernel, channels]` tensor and
/// `dynamic` the `[groups * kernel]` projection of the layer input this row was projected from, group
/// major. A group is one KV head.
pub fn short_conv_dynamic(
    x: &[f64],
    history: &mut ConvHistory,
    static_taps: &[f32],
    dynamic: &[f64],
    kernel: usize,
    groups: usize,
) -> Vec<f64> {
    let channels = x.len();
    let per_group = channels / groups;
    let divisor = ((2 * kernel + 1) as f64).sqrt();
    let mut out = vec![0.0; channels];
    for (channel, value) in out.iter_mut().enumerate() {
        let group = channel / per_group;
        let mut sum = 0.0;
        for lag in 0..kernel {
            let input = if lag == 0 {
                x[channel]
            } else {
                history.back(lag).map_or(0.0, |row| row[channel])
            };
            let tap =
                f64::from(static_taps[lag * channels + channel]) + dynamic[group * kernel + lag];
            sum += input * tap;
        }
        *value = (sum + x[channel]) / divisor;
    }
    history.push(x, kernel);
    out
}

/// Geometry and position rules of one relative-bias attention call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttentionShape {
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// `Some(w)`: a key at distance `>= w` is not attended (the query sees itself and `w - 1` back).
    pub window: Option<usize>,
    /// Columns of the bias table.
    pub extent: usize,
    /// A key at distance `>= extent` is masked (`true`) or attended with a zero bias.
    pub mask_beyond_extent: bool,
}

/// The K and V rows a slot holds: `[rows, kv_heads, head_dim]` each, with the position of every
/// row.
#[derive(Debug, Clone, Copy)]
pub struct CachedRows<'a> {
    pub positions: &'a [u64],
    pub keys: &'a [f64],
    pub values: &'a [f64],
}

/// One query row against the cached rows of its slot.
///
/// `query` is `[query_heads, head_dim]` after its clip (and log scale); `bias` is
/// `[query_heads, extent]`, the query's relative vector through the bias table; `sink` is one logit
/// per head. Cached rows after the query are skipped.
///
/// `p_j = exp(s_j) / (exp(sink) + sum exp(s))` with
/// `s_j = q . k_j / sqrt(head_dim) + bias[position - position_j]`; the sink has no value row, so
/// it only absorbs probability.
pub fn attend(
    shape: &AttentionShape,
    query: &[f64],
    bias: &[f64],
    sink: &[f64],
    position: u64,
    cached: CachedRows<'_>,
) -> Vec<f64> {
    let CachedRows {
        positions,
        keys,
        values,
    } = cached;
    let (heads, kv_heads, dim) = (shape.query_heads, shape.kv_heads, shape.head_dim);
    let kv_width = kv_heads * dim;
    let scale = (dim as f64).powf(-0.5);
    let mut out = vec![0.0; heads * dim];
    let mut scores: Vec<(usize, f64)> = Vec::with_capacity(positions.len());
    for head in 0..heads {
        let kv_head = head / (heads / kv_heads);
        let q = &query[head * dim..(head + 1) * dim];
        scores.clear();
        for (row, &key_position) in positions.iter().enumerate() {
            let Some(distance) = position.checked_sub(key_position) else {
                continue;
            };
            let distance = distance as usize;
            if shape.window.is_some_and(|window| distance >= window) {
                continue;
            }
            let bias = if distance < shape.extent {
                bias[head * shape.extent + distance]
            } else if shape.mask_beyond_extent {
                continue;
            } else {
                0.0
            };
            let key = &keys[row * kv_width + kv_head * dim..][..dim];
            scores.push((row, dot(q, key) * scale + bias));
        }
        let top = scores
            .iter()
            .fold(sink[head], |top, (_, score)| top.max(*score));
        let mut denominator = 0.0;
        for (_, score) in &mut scores {
            *score = (*score - top).exp();
            denominator += *score;
        }
        denominator += (sink[head] - top).exp();
        let target = &mut out[head * dim..(head + 1) * dim];
        for (row, weight) in &scores {
            let probability = weight / denominator;
            let value = &values[row * kv_width + kv_head * dim..][..dim];
            for (o, v) in target.iter_mut().zip(value) {
                *o += probability * v;
            }
        }
    }
    out
}

/// Depth mix of the block attention residual and of the loop message: for each query,
/// `softmax(q . rms(s) / sqrt(hidden))` over `states` (earlier block outputs, then the running
/// stream), summing the RMS-normalized states.
pub fn depth_mix(queries: &[f32], states: &[&[f64]], epsilon: f64) -> Vec<Vec<f64>> {
    let hidden = states[0].len();
    let scale = (hidden as f64).powf(-0.5);
    let normalized: Vec<Vec<f64>> = states
        .iter()
        .map(|state| rms_norm(state, None, epsilon))
        .collect();
    queries
        .chunks_exact(hidden)
        .map(|query| {
            let scores: Vec<f64> = normalized
                .iter()
                .map(|state| dot_f32(query, state) * scale)
                .collect();
            let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|score| (score - top).exp()).collect();
            let denominator: f64 = weights.iter().sum();
            let mut out = vec![0.0; hidden];
            for (weight, state) in weights.iter().zip(&normalized) {
                let probability = weight / denominator;
                for (o, v) in out.iter_mut().zip(state) {
                    *o += probability * v;
                }
            }
            out
        })
        .collect()
}

/// Loop controller: `g = down(cat(rms(x), rms(m)))`,
/// `(1 - tanh(forget g)) * x + tanh(gain g) * m`.
pub fn controller(
    x: &[f64],
    message: &[f64],
    down: &[f32],
    forget: &[f32],
    gain: &[f32],
    epsilon: f64,
) -> Vec<f64> {
    let mut pair = rms_norm(x, None, epsilon);
    pair.extend(rms_norm(message, None, epsilon));
    let gate = linear(down, &pair);
    let forget = linear(forget, &gate);
    let gain = linear(gain, &gate);
    (0..x.len())
        .map(|i| (1.0 - forget[i].tanh()) * x[i] + gain[i].tanh() * message[i])
        .collect()
}

/// Sigmoid router: the top `top_k` experts by `sigmoid(W x) + bias`, ties to the lower expert
/// index, in rank order; each weight is the expert's unbiased sigmoid over the selected sum,
/// times `scale`.
pub fn route(
    x: &[f64],
    weight: &[f32],
    bias: Option<&[f32]>,
    top_k: usize,
    scale: f64,
) -> (Vec<usize>, Vec<f64>) {
    let scores: Vec<f64> = linear(weight, x).into_iter().map(sigmoid).collect();
    let keyed: Vec<f64> = match bias {
        Some(bias) => scores
            .iter()
            .zip(bias)
            .map(|(score, bias)| score + f64::from(*bias))
            .collect(),
        None => scores.clone(),
    };
    let mut order: Vec<usize> = (0..scores.len()).collect();
    // stable: equal keys keep ascending expert order
    order.sort_by(|a, b| keyed[*b].total_cmp(&keyed[*a]));
    order.truncate(top_k);
    let total: f64 = order.iter().map(|&expert| scores[expert]).sum();
    let weights = order
        .iter()
        .map(|&expert| scores[expert] / total * scale)
        .collect();
    (order, weights)
}

/// One routed expert on float inputs: gate/up projection, softcapped SwiGLU, the routing weight,
/// down projection.
///
/// `gate_up` is `[2 * width, hidden]` and `down` is `[hidden, width]`, both in checkpoint row order.
/// `interleaved`: row `2j` is gate `j` and row `2j + 1` is up `j`; otherwise the gate rows come
/// first. `weight_before_down`: the routing weight multiplies the activation, not the output.
pub fn expert_float(
    x: &[f64],
    gate_up: &[f64],
    down: &[f64],
    routing_weight: f64,
    caps: (f64, f64),
    interleaved: bool,
    weight_before_down: bool,
) -> Vec<f64> {
    let hidden = x.len();
    let width = gate_up.len() / hidden / 2;
    let projected: Vec<f64> = gate_up
        .chunks_exact(hidden)
        .map(|row| dot(row, x))
        .collect();
    let activation: Vec<f64> = (0..width)
        .map(|j| {
            let (gate, up) = if interleaved {
                (projected[2 * j], projected[2 * j + 1])
            } else {
                (projected[j], projected[width + j])
            };
            let value = softcapped_swiglu(gate, up, caps.0, caps.1);
            if weight_before_down {
                value * routing_weight
            } else {
                value
            }
        })
        .collect();
    down.chunks_exact(width)
        .map(|row| {
            let value = dot(row, &activation);
            if weight_before_down {
                value
            } else {
                value * routing_weight
            }
        })
        .collect()
}

/// One layer's stacked expert bank as the checkpoint stores it.
#[derive(Debug, Clone, PartialEq)]
pub enum ExpertBank {
    /// e2m1 codes, two per byte with the even element in the low nibble; one e4m3 scale per 16
    /// inputs per row; a per-expert `amax` over one denominator.
    Nvfp4 {
        codes: Vec<u8>,
        scales: Vec<u8>,
        amax: Vec<f32>,
        denominator: f32,
        rows: usize,
        inputs: usize,
    },
    /// e4m3 values with one ue8m0 exponent per 32 inputs per row.
    Mxfp8 {
        values: Vec<u8>,
        exponents: Vec<u8>,
        rows: usize,
        inputs: usize,
    },
}

impl ExpertBank {
    pub fn rows(&self) -> usize {
        match self {
            Self::Nvfp4 { rows, .. } | Self::Mxfp8 { rows, .. } => *rows,
        }
    }

    pub fn inputs(&self) -> usize {
        match self {
            Self::Nvfp4 { inputs, .. } | Self::Mxfp8 { inputs, .. } => *inputs,
        }
    }

    /// An e4m3 NaN code (`0x7f`, `0xff`) anywhere a value is decoded from.
    pub fn has_nan_code(&self) -> bool {
        let nan = |bytes: &[u8]| bytes.iter().any(|byte| byte & 0x7f == 0x7f);
        match self {
            Self::Nvfp4 { scales, .. } => nan(scales),
            Self::Mxfp8 { values, .. } => nan(values),
        }
    }

    /// NVFP4 only: `val(code) * e4m3(scale)` of element `index` of `expert`, exact in f32 (three
    /// and four significand bits). The even element of a byte is its low nibble.
    pub fn code_scale(&self, expert: usize, index: usize) -> Option<f32> {
        match self {
            Self::Nvfp4 {
                codes,
                scales,
                rows,
                inputs,
                ..
            } => {
                let element = expert * rows * inputs + index;
                let byte = codes[element / 2];
                let nibble = if element.is_multiple_of(2) {
                    byte & 0xf
                } else {
                    byte >> 4
                };
                Some(e2m1_decode(nibble) * e4m3_decode(scales[element / 16]))
            }
            Self::Mxfp8 { .. } => None,
        }
    }

    /// One expert's `[rows, inputs]` weights, dequantized exactly.
    ///
    /// NVFP4: `(val(code) * e4m3(scale)) * amax / denominator`; the product in parentheses is exact in
    /// f32 and its product with `amax` is exact in f64, so the only rounding is one f64 division.
    /// MXFP8: `e4m3(value) * 2^(exponent - 127)`, exact.
    pub fn expert(&self, expert: usize) -> Vec<f64> {
        match self {
            Self::Nvfp4 {
                amax,
                denominator,
                rows,
                inputs,
                ..
            } => {
                let amax = f64::from(amax[expert]);
                let denominator = f64::from(*denominator);
                (0..rows * inputs)
                    .map(|index| {
                        let scaled = self.code_scale(expert, index).expect("an NVFP4 bank");
                        f64::from(scaled) * amax / denominator
                    })
                    .collect()
            }
            Self::Mxfp8 {
                values,
                exponents,
                rows,
                inputs,
            } => {
                let per_expert = rows * inputs;
                (0..per_expert)
                    .map(|index| {
                        let element = expert * per_expert + index;
                        let exponent = i32::from(exponents[element / 32]) - 127;
                        f64::from(e4m3_decode(values[element])) * 2f64.powi(exponent)
                    })
                    .collect()
            }
        }
    }
}
