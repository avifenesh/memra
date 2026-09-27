//! Source-only V codec diagnostic. The capture is the F32 value output of
//! MiMo's QKV gather, including its pinned 0.707 scale, before any KV append.
//! Candidate bytes are encoded and decoded here; attention still uses the
//! existing cache format. Component errors do not establish task quality.

use std::path::Path;
use std::sync::Arc;

use cudarc::driver::CudaContext;
use memra_gguf::nvfp4_repack::{KVALUES_MXFP4, dequant_gguf_row, f32_to_nvfp4, ue4m3_to_f32};
use memra_gguf::source::SafetensorsSource;

use super::{Engine, Fail, LAYERS, MiMoTextWeights};
use crate::mimo_attn_load::MiMoAttentionGeometry;

const TOKENS: usize = 128;
const HEADS: usize = 4;
const VALUE_WIDTH: usize = 128;
const ROW_WIDTH: usize = HEADS * VALUE_WIDTH;
const GGUF_ROW_BYTES: usize = HEADS * (VALUE_WIDTH / 64) * 36;
const FP4_G8_ROW_BYTES: usize = HEADS * (VALUE_WIDTH / 8) * 5;
const S5_G16_ROW_BYTES: usize = HEADS * (VALUE_WIDTH / 16) * 11;
const FOUR_GIB: usize = 4 * 1024 * 1024 * 1024;
const FORMATS: [(&str, usize); 3] = [
    ("gguf_nvfp4_g16", GGUF_ROW_BYTES),
    ("fp4_ue4m3_g8", FP4_G8_ROW_BYTES),
    ("signed_uniform5_ue4m3_g16", S5_G16_ROW_BYTES),
];
const FP4_MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

// Match the pinned NVFP4 host encoder's positive UE4M3 scale search:
// nearest representable positive value, with the lower code winning ties.
fn scale_byte(target: f32) -> u8 {
    if !target.is_finite() || target <= 0.0 {
        return 0;
    }
    let mut best = 0u8;
    let mut distance = f32::INFINITY;
    for code in 1u8..0x7f {
        let error = (ue4m3_to_f32(code) * 2.0 - target).abs();
        if error < distance {
            best = code;
            distance = error;
        }
    }
    best
}

// FP4 nearest magnitude; ties go to the smaller magnitude. Zero is canonical
// even for negative input, and nonzero negative codes use the GGUF sign bit.
fn fp4_code(value: f32, inverse_scale: f32) -> u8 {
    let scaled = (value * inverse_scale).abs();
    let mut best = 0usize;
    let mut distance = f32::INFINITY;
    for (code, &magnitude) in FP4_MAGNITUDES.iter().enumerate() {
        let error = (scaled - magnitude).abs();
        if error < distance {
            best = code;
            distance = error;
        }
    }
    if best == 0 {
        0
    } else if value.is_sign_negative() {
        best as u8 | 8
    } else {
        best as u8
    }
}

// One scale byte plus four sequential low/high-nibble pairs per 8 values.
fn encode_fp4_g8(row: &[f32]) -> Vec<u8> {
    assert_eq!(row.len(), ROW_WIDTH);
    let mut encoded = Vec::with_capacity(FP4_G8_ROW_BYTES);
    for group in row.chunks_exact(8) {
        let amax = group.iter().fold(0.0f32, |max, &x| max.max(x.abs()));
        let scale = scale_byte(amax / 6.0);
        let raw_scale = ue4m3_to_f32(scale) * 2.0;
        let inverse_scale = if raw_scale > 0.0 {
            1.0 / raw_scale
        } else {
            0.0
        };
        encoded.push(scale);
        for pair in group.chunks_exact(2) {
            encoded
                .push(fp4_code(pair[0], inverse_scale) | (fp4_code(pair[1], inverse_scale) << 4));
        }
    }
    encoded
}

fn decode_fp4_g8(encoded: &[u8]) -> Vec<f32> {
    assert_eq!(encoded.len(), FP4_G8_ROW_BYTES);
    let mut decoded = Vec::with_capacity(ROW_WIDTH);
    for group in encoded.chunks_exact(5) {
        // KVALUES_MXFP4 is doubled; ue4m3_to_f32 halves the raw scale.
        let half_scale = ue4m3_to_f32(group[0]);
        for &packed in &group[1..] {
            decoded.push(KVALUES_MXFP4[(packed & 15) as usize] as f32 * half_scale);
            decoded.push(KVALUES_MXFP4[(packed >> 4) as usize] as f32 * half_scale);
        }
    }
    decoded
}

// Signed two's-complement codes [-16, 15] use all 32 patterns. Scale is
// max(positive/15, abs(negative)/16), then nearest positive UE4M3 scale.
// Codes round to nearest with exact half ties away from zero, then clamp.
// Each group is one scale byte plus 80 little-endian packed code bits.
fn encode_s5_g16(row: &[f32]) -> Vec<u8> {
    assert_eq!(row.len(), ROW_WIDTH);
    let mut encoded = Vec::with_capacity(S5_G16_ROW_BYTES);
    for group in row.chunks_exact(16) {
        let positive = group.iter().fold(0.0f32, |max, &x| max.max(x));
        let negative = group.iter().fold(0.0f32, |max, &x| max.max(-x));
        let scale = scale_byte((positive / 15.0).max(negative / 16.0));
        let raw_scale = ue4m3_to_f32(scale) * 2.0;
        let mut block = [0u8; 11];
        block[0] = scale;
        for (index, &value) in group.iter().enumerate() {
            let code = if raw_scale > 0.0 {
                (value / raw_scale).round().clamp(-16.0, 15.0) as i8
            } else {
                0
            };
            let bits = (code as u8) & 31;
            let bit = index * 5;
            let byte = 1 + bit / 8;
            let shift = bit % 8;
            block[byte] |= bits << shift;
            if shift > 3 {
                block[byte + 1] |= bits >> (8 - shift);
            }
        }
        encoded.extend_from_slice(&block);
    }
    encoded
}

fn decode_s5_g16(encoded: &[u8]) -> Vec<f32> {
    assert_eq!(encoded.len(), S5_G16_ROW_BYTES);
    let mut decoded = Vec::with_capacity(ROW_WIDTH);
    for group in encoded.chunks_exact(11) {
        let raw_scale = ue4m3_to_f32(group[0]) * 2.0;
        for index in 0..16 {
            let bit = index * 5;
            let byte = 1 + bit / 8;
            let shift = bit % 8;
            let mut packed = u16::from(group[byte]);
            if shift > 3 {
                packed |= u16::from(group[byte + 1]) << 8;
            }
            let bits = i32::from((packed >> shift) & 31);
            let code = if bits >= 16 { bits - 32 } else { bits };
            decoded.push(code as f32 * raw_scale);
        }
    }
    decoded
}

#[derive(Clone, Copy, Default)]
struct ErrorStats {
    squared_error: f64,
    squared_source: f64,
    max_error: f64,
    decoded_zeros: usize,
    samples: usize,
}

impl ErrorStats {
    fn observe(&mut self, source: &[f32], decoded: &[f32]) -> Result<(), &'static str> {
        if source.len() != ROW_WIDTH
            || decoded.len() != source.len()
            || source.iter().any(|value| !value.is_finite())
            || decoded.iter().any(|value| !value.is_finite())
        {
            return Err("MiMo V codec probe has invalid F32 rows");
        }
        for (&want, &got) in source.iter().zip(decoded) {
            let error = (f64::from(got) - f64::from(want)).abs();
            self.squared_error += error * error;
            self.squared_source += f64::from(want).powi(2);
            self.max_error = self.max_error.max(error);
            self.decoded_zeros += usize::from(got == 0.0);
        }
        self.samples += source.len();
        Ok(())
    }

    fn merge(&mut self, other: Self) {
        self.squared_error += other.squared_error;
        self.squared_source += other.squared_source;
        self.max_error = self.max_error.max(other.max_error);
        self.decoded_zeros += other.decoded_zeros;
        self.samples += other.samples;
    }

    fn relative_l2(self) -> Result<f64, &'static str> {
        if self.samples == 0 || self.squared_source == 0.0 {
            return Err("MiMo V codec probe has no nonzero source activations");
        }
        Ok((self.squared_error / self.squared_source).sqrt())
    }

    fn zero_fraction(self) -> f64 {
        self.decoded_zeros as f64 / self.samples as f64
    }
}

fn report(scope: &str, layer: &str, stats: &[ErrorStats; 3]) -> Result<(), Fail> {
    for ((name, bytes), metric) in FORMATS.iter().zip(stats) {
        println!(
            "metric\t{scope}\t{layer}\t{name}\t{bytes}\t{}\t{:.9e}\t{:.9e}\t{:.9e}",
            metric.samples,
            metric.relative_l2()?,
            metric.max_error,
            metric.zero_fraction(),
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires pinned MiMo source and exactly two dedicated RTX PRO 6000 Blackwell cards"]
fn first_chunk_value_codec_quality_probe() -> Result<(), Fail> {
    let root = std::env::var("MIMO_PINNED_SOURCE_ROOT")?;
    if CudaContext::device_count()? != 2 {
        return Err("MiMo V codec probe requires exactly two visible GPUs".into());
    }
    let cards = [Engine::new(0)?, Engine::new(1)?];
    let engines = [&cards[0], &cards[1]];
    let names = [cards[0].ctx().name()?, cards[1].ctx().name()?];
    if cards[0].stream().context().ordinal() == cards[1].stream().context().ordinal()
        || names
            .iter()
            .any(|name| !name.contains("RTX PRO 6000") || !name.contains("Blackwell"))
    {
        return Err("MiMo V codec probe needs two distinct RTX PRO 6000 Blackwell cards".into());
    }
    let source = Arc::new(SafetensorsSource::open(Path::new(&root))?);
    let text = MiMoTextWeights::load(engines, source)?;
    let ids = (42..42 + TOKENS as u32).collect::<Vec<_>>();
    let prepared = text.modal_embedding_gpu_chunk(&cards[0], &ids, &[], &[], &[])?;
    if prepared.token_count() != TOKENS || prepared.requires_payload_identity() {
        return Err("MiMo V codec probe lost its 128 source text token rows".into());
    }
    let mut sequence = text.compressed_text_forward(engines, TOKENS, [FOUR_GIB; 2])?;
    sequence.value_probe = Some(vec![None; LAYERS]);
    let step = sequence.consume_embedding_chunk_batched(&prepared)?;
    if step.position != TOKENS - 1 || sequence.position() != TOKENS {
        return Err("MiMo V codec probe did not finish the first source text chunk".into());
    }
    let captured = sequence
        .value_probe
        .take()
        .ok_or("MiMo V codec probe capture disappeared")?;
    let mut pooled = [ErrorStats::default(); 3];
    let mut global_layers = 0;
    let mut source_zeros = 0;
    println!("format\tmimo_first_chunk_value_codec_quality_v1");
    println!(
        "target_source\tXiaomiMiMo/MiMo-V2.6-Flash-RL@3b38d063180c3e4aed9691fdc735f3d10b266ee4"
    );
    println!("gpu\t{}\t{}", names[0], names[1]);
    println!("shape\t128_tokens\t4_kv_heads\t128_values_per_head\tpre_cache_f32_v");
    println!("token_ids\t42..170_exclusive");
    println!(
        "columns\tscope\tlayer\tcodec\tbytes_per_4x128_row\tsamples\trelative_l2\tmax_abs_error\tdecoded_zero_fraction"
    );
    for (layer, maybe_values) in captured.into_iter().enumerate() {
        let geometry = MiMoAttentionGeometry::from_plan(&text.plan, layer)?;
        if geometry.window != 0 {
            if maybe_values.is_some() {
                return Err("MiMo V codec probe captured a local layer".into());
            }
            continue;
        }
        let values = maybe_values.ok_or("MiMo V codec probe missed a global layer")?;
        if values.len() != TOKENS * ROW_WIDTH {
            return Err("MiMo V codec probe global layer row count changed".into());
        }
        let mut layer_stats = [ErrorStats::default(); 3];
        let input_zeros = values.iter().filter(|&&value| value == 0.0).count();
        source_zeros += input_zeros;
        global_layers += 1;
        println!(
            "source_zeros\tlayer\t{layer}\t{:.9e}",
            input_zeros as f64 / values.len() as f64
        );
        for row in values.chunks_exact(ROW_WIDTH) {
            let baseline = f32_to_nvfp4(row);
            let fp4_g8 = encode_fp4_g8(row);
            let s5_g16 = encode_s5_g16(row);
            if baseline.len() != GGUF_ROW_BYTES
                || fp4_g8.len() != FP4_G8_ROW_BYTES
                || s5_g16.len() != S5_G16_ROW_BYTES
            {
                return Err("MiMo V codec probe encoded row bytes changed".into());
            }
            let decoded = [
                dequant_gguf_row(&baseline, ROW_WIDTH),
                decode_fp4_g8(&fp4_g8),
                decode_s5_g16(&s5_g16),
            ];
            for (metric, actual) in layer_stats.iter_mut().zip(&decoded) {
                metric.observe(row, actual)?;
            }
        }
        report("layer", &layer.to_string(), &layer_stats)?;
        for (pooled_metric, layer_metric) in pooled.iter_mut().zip(layer_stats) {
            pooled_metric.merge(layer_metric);
        }
    }
    if global_layers != 9 || pooled[0].samples != global_layers * TOKENS * ROW_WIDTH {
        return Err("MiMo V codec probe global layer census changed".into());
    }
    println!(
        "source_zeros\tpooled\tall\t{:.9e}",
        source_zeros as f64 / pooled[0].samples as f64
    );
    report("pooled", "all", &pooled)?;
    Ok(())
}
