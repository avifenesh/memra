//! Layer-streamed, two-device MiMo source text-token diagnostic.
//! This is an offline arithmetic rung, not resident serving or model support.

use std::fmt::Write as _;
use std::io::Write as _;
use std::path::Path;
use std::time::Instant;

use cudarc::driver::CudaSlice;
use memra_engine::Engine;
use memra_engine::QT_F8_E4M3_BLK;
use memra_engine::mimo_attn_load::MiMoSourceAttention;
use memra_engine::mimo_mixed_attn_ffi::MiMoMixedAttentionWorkspace;
use memra_engine::mimo_source_moe::{
    GroupedMiMoMoeLayer, PinnedMiMoSource, ResidentMiMoMoeLayer, source_moe_token,
};
use memra_engine::model::GpuTensor;
use memra_gguf::GgmlType;
use memra_gguf::checkpoint_binding::{CheckpointBinding, RecordingSource};
use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::dequant::{dequantize, fp16_to_f32};
use memra_gguf::model_packs;
use memra_gguf::model_plan::{
    ActivationPlan, AttentionPlan, FullAttentionPlan, LayerPlan, MlpPlan, ModelPlan, NormKind,
    ResidualTopology, RopeFactors, StatePlan, WeightTransform,
};
use memra_gguf::nvfp4_repack::{f32_to_fp8_e4m3, f32_to_nvfp4, f32_to_q8_0, fp8_e4m3_to_f32};
use memra_gguf::safetensors::StModel;
use memra_gguf::source::{SafetensorsSource, TensorSource};
use sha2::{Digest, Sha256};

type Fail = Box<dyn std::error::Error>;

const CONFIG_SHA256: &str = "61bea4a0f7a0dd8969f8cae528761e26b697dd12ff63e98804c3f0945492e621";
const REVISION: &str = "3b38d063180c3e4aed9691fdc735f3d10b266ee4";
const HIDDEN: usize = 4096;
const VOCAB: usize = 152576;
const LAYERS: usize = 48;
const STAGE_CUT: usize = 24;
const MAX_DIAGNOSTIC_TOKENS: usize = 256;
const QK: usize = 192;
const VALUE: usize = 128;
const GLOBAL_LAYERS: [usize; 9] = [0, 5, 11, 17, 23, 29, 35, 41, 47];

fn decode_bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16))
        .collect()
}

fn read_vector(model: &StModel, name: &str, length: usize) -> Result<Vec<f32>, Fail> {
    let (info, bytes) = model.raw(name).ok_or_else(|| format!("{name}: missing"))?;
    if info.shape != [length as u64] {
        return Err(format!("{name}: wrong shape {:?}", info.shape).into());
    }
    let values = match info.dtype.as_str() {
        "BF16" if bytes.len() == length * 2 => decode_bf16(bytes),
        "F32" if bytes.len() == length * 4 => bytes
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]]))
            .collect(),
        _ => return Err(format!("{name}: wrong dtype or byte length").into()),
    };
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{name}: non-finite value").into());
    }
    Ok(values)
}

fn embedding_row(model: &StModel, token: usize) -> Result<Vec<f32>, Fail> {
    let name = "model.embed_tokens.weight";
    let (info, bytes) = model.raw(name).ok_or("source embedding is missing")?;
    if info.dtype != "BF16"
        || info.shape != [VOCAB as u64, HIDDEN as u64]
        || bytes.len() != VOCAB * HIDDEN * 2
        || token >= VOCAB
    {
        return Err("source embedding storage or token id changed".into());
    }
    let start = token * HIDDEN * 2;
    let values = decode_bf16(&bytes[start..start + HIDDEN * 2]);
    if values.iter().any(|value| !value.is_finite()) {
        return Err("source embedding contains a non-finite value".into());
    }
    Ok(values)
}

fn bf16_matrix(
    engine: &Engine,
    model: &StModel,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<GpuTensor, Fail> {
    let (info, bytes) = model.raw(name).ok_or_else(|| format!("{name}: missing"))?;
    if info.dtype != "BF16"
        || info.shape != [rows as u64, cols as u64]
        || bytes.len() != rows * cols * 2
    {
        return Err(format!("{name}: BF16 matrix geometry changed").into());
    }
    Ok(GpuTensor::FloatBf16 {
        data: engine.htod_bytes(bytes)?,
        ne: vec![cols as u64, rows as u64],
    })
}

fn f32_mirror_of_bf16(engine: &Engine, tensor: GpuTensor) -> Result<GpuTensor, Fail> {
    let GpuTensor::FloatBf16 { data, ne } = tensor else {
        return Err("MiMo f32 mirror requires a source BF16 matrix".into());
    };
    if ne.len() != 2 || data.len() != ne[0] as usize * ne[1] as usize * 2 {
        return Err("MiMo f32 mirror input geometry changed".into());
    }
    let values = engine.bf16_to_f32(&data.slice(..), data.len() / 2)?;
    engine.stream().synchronize()?;
    Ok(GpuTensor::Float { data: values, ne })
}

fn normalized(
    engine: &Engine,
    model: &StModel,
    input: &CudaSlice<f32>,
    name: &str,
    norm: &memra_gguf::model_plan::NormPlan,
) -> Result<CudaSlice<f32>, Fail> {
    let weight = engine.htod(&read_vector(model, name, HIDDEN)?)?;
    normalized_with_weight(engine, input, &weight, name, norm)
}

fn normalized_with_weight(
    engine: &Engine,
    input: &CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    name: &str,
    norm: &memra_gguf::model_plan::NormPlan,
) -> Result<CudaSlice<f32>, Fail> {
    if norm.kind != NormKind::Rms
        || norm.weight_transform != WeightTransform::Identity
        || input.len() != HIDDEN
        || weight.len() != HIDDEN
        || weight.ordinal() != engine.stream().context().ordinal()
    {
        return Err(format!("{name}: unsupported MiMo norm").into());
    }
    let mut output = engine.uninit(HIDDEN)?;
    engine.rms_norm(input, weight, &mut output, HIDDEN, 1, norm.epsilon)?;
    Ok(output)
}

fn checked_attention(plan: &LayerPlan) -> Result<(&FullAttentionPlan, bool), Fail> {
    let layer = plan.index as usize;
    let global = GLOBAL_LAYERS.contains(&layer);
    let attention = match (&plan.attention, global) {
        (AttentionPlan::Full(full), true) if full.kv_heads == 4 => full,
        (AttentionPlan::SlidingWindow { attention, window }, false)
            if *window == 128 && attention.kv_heads == 8 =>
        {
            attention
        }
        _ => return Err(format!("layer {layer}: unsupported MiMo attention").into()),
    };
    let state_is_pinned = matches!(
        (&plan.state, global),
        (
            StatePlan::KvCache {
                key_width: 768,
                value_width: 512,
            },
            true,
        ) | (
            StatePlan::SlidingKvCache {
                key_width: 1536,
                value_width: 1024,
                window: 128,
            },
            false,
        )
    );
    if attention.query_heads != 64
        || attention.key_head_dim as usize != QK
        || attention.value_head_dim as usize != VALUE
        || attention.rope.dimensions != 64
        || !matches!(attention.rope.factors, RopeFactors::None)
        || !state_is_pinned
        || plan.residual != ResidualTopology::Serial
        || plan.sparse_overlay.is_some()
        || plan.ple.is_some()
    {
        return Err(format!("layer {layer}: changed attention geometry or RoPE").into());
    }
    Ok((attention, global))
}

struct KvState {
    key: CudaSlice<f32>,
    value: CudaSlice<f32>,
    tokens: usize,
}

struct PackedKvState {
    key: CudaSlice<u8>,
    value: CudaSlice<u8>,
    tokens: usize,
}

struct AttentionCache<'a> {
    plain: &'a mut Option<KvState>,
    packed: &'a mut Option<PackedKvState>,
    mixed_workspace: Option<&'a mut MiMoMixedAttentionWorkspace>,
}

fn record_phase(
    report: &mut String,
    engine: &Engine,
    enabled: bool,
    turn: usize,
    layer: usize,
    name: &str,
    started: Instant,
) -> Result<(), Fail> {
    if enabled {
        engine.stream().synchronize()?;
        writeln!(
            report,
            "phase_ms\t{turn}\t{layer}\t{name}\t{:.3}",
            started.elapsed().as_secs_f64() * 1000.0
        )?;
    }
    Ok(())
}

struct AttentionPhase<'a> {
    report: &'a mut String,
    enabled: bool,
    turn: usize,
    resident: Option<&'a ResidentTextLayer>,
    fp8_probe: bool,
    fp8_replay: bool,
    q8q5_probe: bool,
    q8q5_replay: bool,
    q8q8_probe: bool,
    q8q8_replay: bool,
    global_only_kv: bool,
    cpu_pair: Option<CpuKvPair>,
    mixed_gpu: bool,
}

impl AttentionPhase<'_> {
    fn record(
        &mut self,
        engine: &Engine,
        layer: usize,
        name: &str,
        started: Instant,
    ) -> Result<(), Fail> {
        record_phase(
            &mut *self.report,
            engine,
            self.enabled,
            self.turn,
            layer,
            name,
            started,
        )
    }
}

struct Fp8RowStats {
    max_abs: f32,
    max_error: f32,
    rms_error: f32,
    saturated: usize,
    zeroed: usize,
    cpu_code_mismatches: usize,
}

struct RestoredKv {
    key: CudaSlice<f32>,
    value: CudaSlice<f32>,
}

fn fp8_row_stats(original: &[f32], gpu_codes: &[u8]) -> Result<(Fp8RowStats, Vec<f32>), Fail> {
    if original.len() != gpu_codes.len()
        || original.is_empty()
        || original.iter().any(|value| !value.is_finite())
    {
        return Err("MiMo FP8 KV row is non-finite or mismatched".into());
    }
    let mut max_abs = 0.0f32;
    let mut max_error = 0.0f32;
    let mut squared_error = 0.0f64;
    let mut saturated = 0usize;
    let mut zeroed = 0usize;
    let mut cpu_code_mismatches = 0usize;
    let mut decoded = Vec::with_capacity(original.len());
    for (&value, &code) in original.iter().zip(gpu_codes) {
        let restored = fp8_e4m3_to_f32(code);
        let error = (restored - value).abs();
        max_abs = max_abs.max(value.abs());
        max_error = max_error.max(error);
        squared_error += (error as f64).powi(2);
        saturated += usize::from(value.abs() > 448.0);
        zeroed += usize::from(value != 0.0 && restored == 0.0);
        cpu_code_mismatches += usize::from(f32_to_fp8_e4m3(value) != code);
        decoded.push(restored);
    }
    Ok((
        Fp8RowStats {
            max_abs,
            max_error,
            rms_error: (squared_error / original.len() as f64).sqrt() as f32,
            saturated,
            zeroed,
            cpu_code_mismatches,
        },
        decoded,
    ))
}

fn fp8_kv_probe(
    engine: &Engine,
    key: &CudaSlice<f32>,
    value: &CudaSlice<f32>,
    phase: &mut AttentionPhase<'_>,
    layer: usize,
) -> Result<Option<RestoredKv>, Fail> {
    if !phase.fp8_probe {
        return Ok(None);
    }
    let mut key_codes = engine.alloc_u8(key.len())?;
    let mut value_codes = engine.alloc_u8(value.len())?;
    engine.append_kv_quantized(
        key,
        value,
        &mut key_codes,
        &mut value_codes,
        0,
        key.len(),
        value.len(),
        key.len(),
        value.len(),
        true,
    )?;
    let key_original = engine.dtoh(key)?;
    let value_original = engine.dtoh(value)?;
    let key_encoded = engine.dtoh_u8(&key_codes)?;
    let value_encoded = engine.dtoh_u8(&value_codes)?;
    for (name, original, encoded) in [
        ("key", &key_original, &key_encoded),
        ("value", &value_original, &value_encoded),
    ] {
        let (stats, _) = fp8_row_stats(original, encoded)?;
        writeln!(
            phase.report,
            "kv_fp8_probe\t{}\t{layer}\t{name}\t{}\t{:.9e}\t{:.9e}\t{:.9e}\t{}\t{}\t{}",
            phase.turn,
            original.len(),
            stats.max_abs,
            stats.max_error,
            stats.rms_error,
            stats.saturated,
            stats.zeroed,
            stats.cpu_code_mismatches
        )?;
    }
    if phase.fp8_replay {
        let (_, key_restored) = fp8_row_stats(&key_original, &key_encoded)?;
        let (_, value_restored) = fp8_row_stats(&value_original, &value_encoded)?;
        Ok(Some(RestoredKv {
            key: engine.htod(&key_restored)?,
            value: engine.htod(&value_restored)?,
        }))
    } else {
        Ok(None)
    }
}

fn decode_q5_1(raw: &[u8], values: usize) -> Result<Vec<f32>, Fail> {
    if !values.is_multiple_of(32) || raw.len() != values / 32 * 24 {
        return Err("MiMo q5_1 value plane has wrong block extent".into());
    }
    let mut decoded = Vec::with_capacity(values);
    for block in raw.chunks_exact(24) {
        let d = fp16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let m = fp16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let high = u32::from_le_bytes(block[4..8].try_into()?);
        for lane in 0..32 {
            let nibble = if lane < 16 {
                block[8 + lane] & 15
            } else {
                block[8 + lane - 16] >> 4
            };
            let code = u32::from(nibble) | (((high >> lane) & 1) << 4);
            decoded.push(d * code as f32 + m);
        }
    }
    if decoded.iter().any(|value| !value.is_finite()) {
        return Err("MiMo q5_1 value plane decoded non-finite values".into());
    }
    Ok(decoded)
}

fn q8q5_row_stats(original: &[f32], restored: &[f32]) -> Result<(f32, f32, f32, usize), Fail> {
    if original.len() != restored.len()
        || original.is_empty()
        || original
            .iter()
            .chain(restored)
            .any(|value| !value.is_finite())
    {
        return Err("MiMo block-quantized KV row is non-finite or mismatched".into());
    }
    let mut max_abs = 0.0f32;
    let mut max_error = 0.0f32;
    let mut squared_error = 0.0f64;
    let mut zeroed = 0usize;
    for (&value, &decoded) in original.iter().zip(restored) {
        let error = (value - decoded).abs();
        max_abs = max_abs.max(value.abs());
        max_error = max_error.max(error);
        squared_error += (error as f64).powi(2);
        zeroed += usize::from(value != 0.0 && decoded == 0.0);
    }
    Ok((
        max_abs,
        max_error,
        (squared_error / original.len() as f64).sqrt() as f32,
        zeroed,
    ))
}

fn q8q5_kv_probe(
    engine: &Engine,
    key: &CudaSlice<f32>,
    value: &CudaSlice<f32>,
    phase: &mut AttentionPhase<'_>,
    layer: usize,
) -> Result<Option<RestoredKv>, Fail> {
    if !phase.q8q5_probe {
        return Ok(None);
    }
    if !key.len().is_multiple_of(32) || !value.len().is_multiple_of(32) {
        return Err("MiMo q8_0/q5_1 KV row widths must be 32-aligned".into());
    }
    let key_bytes = key.len() / 32 * 34;
    let value_bytes = value.len() / 32 * 24;
    let mut key_codes = engine.alloc_u8(key_bytes)?;
    let mut value_codes = engine.alloc_u8(value_bytes)?;
    engine.append_kv_quantized(
        key,
        value,
        &mut key_codes,
        &mut value_codes,
        0,
        key.len(),
        value.len(),
        key_bytes,
        value_bytes,
        false,
    )?;
    let key_original = engine.dtoh(key)?;
    let value_original = engine.dtoh(value)?;
    let key_encoded = engine.dtoh_u8(&key_codes)?;
    let value_encoded = engine.dtoh_u8(&value_codes)?;
    let key_restored = dequantize(GgmlType::Q8_0, &key_encoded, key.len());
    let value_restored = decode_q5_1(&value_encoded, value.len())?;
    for (name, original, restored) in [
        ("key", &key_original, &key_restored),
        ("value", &value_original, &value_restored),
    ] {
        let (max_abs, max_error, rms_error, zeroed) = q8q5_row_stats(original, restored)?;
        writeln!(
            phase.report,
            "kv_q8q5_probe\t{}\t{layer}\t{name}\t{}\t{max_abs:.9e}\t{max_error:.9e}\t{rms_error:.9e}\t{zeroed}",
            phase.turn,
            original.len()
        )?;
    }
    if phase.q8q5_replay {
        Ok(Some(RestoredKv {
            key: engine.htod(&key_restored)?,
            value: engine.htod(&value_restored)?,
        }))
    } else {
        Ok(None)
    }
}

fn q8q8_kv_probe(
    engine: &Engine,
    key: &CudaSlice<f32>,
    value: &CudaSlice<f32>,
    phase: &mut AttentionPhase<'_>,
    layer: usize,
) -> Result<Option<RestoredKv>, Fail> {
    if !phase.q8q8_probe {
        return Ok(None);
    }
    if !key.len().is_multiple_of(32) || !value.len().is_multiple_of(32) {
        return Err("MiMo q8_0/q8_0 KV widths must be 32-aligned".into());
    }
    let key_bytes = key.len() / 32 * 34;
    let value_bytes = value.len() / 32 * 34;
    let dummy_value_bytes = value.len() / 32 * 24;
    let mut key_codes = engine.alloc_u8(key_bytes)?;
    let mut value_q5_dummy = engine.alloc_u8(dummy_value_bytes)?;
    engine.append_kv_quantized(
        key,
        value,
        &mut key_codes,
        &mut value_q5_dummy,
        0,
        key.len(),
        value.len(),
        key_bytes,
        dummy_value_bytes,
        false,
    )?;
    let mut value_codes = engine.alloc_u8(value_bytes)?;
    let mut second_q5_dummy = engine.alloc_u8(dummy_value_bytes)?;
    engine.append_kv_quantized(
        value,
        value,
        &mut value_codes,
        &mut second_q5_dummy,
        0,
        value.len(),
        value.len(),
        value_bytes,
        dummy_value_bytes,
        false,
    )?;
    let key_original = engine.dtoh(key)?;
    let value_original = engine.dtoh(value)?;
    let key_encoded = engine.dtoh_u8(&key_codes)?;
    let value_encoded = engine.dtoh_u8(&value_codes)?;
    let key_restored = dequantize(GgmlType::Q8_0, &key_encoded, key.len());
    let value_restored = dequantize(GgmlType::Q8_0, &value_encoded, value.len());
    for (name, original, restored) in [
        ("key", &key_original, &key_restored),
        ("value", &value_original, &value_restored),
    ] {
        let (max_abs, max_error, rms_error, zeroed) = q8q5_row_stats(original, restored)?;
        writeln!(
            phase.report,
            "kv_q8q8_probe\t{}\t{layer}\t{name}\t{}\t{max_abs:.9e}\t{max_error:.9e}\t{rms_error:.9e}\t{zeroed}",
            phase.turn,
            original.len()
        )?;
    }
    if phase.q8q8_replay {
        Ok(Some(RestoredKv {
            key: engine.htod(&key_restored)?,
            value: engine.htod(&value_restored)?,
        }))
    } else {
        Ok(None)
    }
}

#[derive(Clone, Copy)]
enum CpuKvFormat {
    Nvfp4,
    Q8,
    Int16,
}

impl CpuKvFormat {
    fn label(self) -> &'static str {
        match self {
            Self::Nvfp4 => "nvfp4_e2m1_scale16",
            Self::Q8 => "q8_0_scale32",
            Self::Int16 => "int16_scale_per_row_f32",
        }
    }
}

#[derive(Clone, Copy)]
struct CpuKvPair {
    key: CpuKvFormat,
    value: CpuKvFormat,
    label: &'static str,
}

fn cpu_kv_pair(name: &str) -> Option<CpuKvPair> {
    use CpuKvFormat::{Int16, Nvfp4, Q8};
    let (key, value, label) = match name {
        "nvfp4-nvfp4" => (Nvfp4, Nvfp4, "nvfp4-nvfp4"),
        "q8-nvfp4" => (Q8, Nvfp4, "q8-nvfp4"),
        "nvfp4-int16" => (Nvfp4, Int16, "nvfp4-int16"),
        "int16-nvfp4" => (Int16, Nvfp4, "int16-nvfp4"),
        "q8-int16" => (Q8, Int16, "q8-int16"),
        "int16-int16" => (Int16, Int16, "int16-int16"),
        _ => return None,
    };
    Some(CpuKvPair { key, value, label })
}

fn cpu_kv_roundtrip(values: &[f32], format: CpuKvFormat) -> Result<(Vec<f32>, usize), Fail> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err("MiMo CPU format candidate row is empty or non-finite".into());
    }
    let result = match format {
        CpuKvFormat::Nvfp4 if values.len().is_multiple_of(64) => {
            let bytes = f32_to_nvfp4(values);
            (
                dequantize(GgmlType::NVFP4, &bytes, values.len()),
                bytes.len(),
            )
        }
        CpuKvFormat::Q8 if values.len().is_multiple_of(32) => {
            let bytes = f32_to_q8_0(values);
            (
                dequantize(GgmlType::Q8_0, &bytes, values.len()),
                bytes.len(),
            )
        }
        CpuKvFormat::Int16 => {
            let max_abs = values
                .iter()
                .fold(0.0f32, |max, value| max.max(value.abs()));
            let scale = if max_abs > 0.0 {
                max_abs / 32767.0
            } else {
                1.0
            };
            let restored = values
                .iter()
                .map(|value| {
                    let code = (value / scale).round_ties_even().clamp(-32767.0, 32767.0) as i16;
                    code as f32 * scale
                })
                .collect();
            (restored, values.len() * 2 + 4)
        }
        _ => return Err("MiMo CPU format candidate row alignment changed".into()),
    };
    Ok(result)
}

fn cpu_kv_pair_probe(
    engine: &Engine,
    key: &CudaSlice<f32>,
    value: &CudaSlice<f32>,
    phase: &mut AttentionPhase<'_>,
    layer: usize,
) -> Result<Option<RestoredKv>, Fail> {
    let Some(pair) = phase.cpu_pair else {
        return Ok(None);
    };
    let key_original = engine.dtoh(key)?;
    let value_original = engine.dtoh(value)?;
    let (key_restored, key_bytes) = cpu_kv_roundtrip(&key_original, pair.key)?;
    let (value_restored, value_bytes) = cpu_kv_roundtrip(&value_original, pair.value)?;
    for (name, format, original, restored, bytes) in [
        ("key", pair.key, &key_original, &key_restored, key_bytes),
        (
            "value",
            pair.value,
            &value_original,
            &value_restored,
            value_bytes,
        ),
    ] {
        let (max_abs, max_error, rms_error, zeroed) = q8q5_row_stats(original, restored)?;
        writeln!(
            phase.report,
            "kv_cpu_pair\t{}\t{layer}\t{name}\t{}\t{}\t{max_abs:.9e}\t{max_error:.9e}\t{rms_error:.9e}\t{zeroed}\t{bytes}",
            phase.turn,
            format.label(),
            original.len()
        )?;
    }
    Ok(Some(RestoredKv {
        key: engine.htod(&key_restored)?,
        value: engine.htod(&value_restored)?,
    }))
}

struct ResidentDense {
    gate: GpuTensor,
    up: GpuTensor,
    down: GpuTensor,
}

struct ResidentTextLayer {
    input_norm: CudaSlice<f32>,
    qkv: Vec<GpuTensor>,
    sink: Option<CudaSlice<f32>>,
    o_proj: GpuTensor,
    post_norm: CudaSlice<f32>,
    dense: Option<ResidentDense>,
}

struct ResidentTextSources<'a> {
    model: &'a StModel,
    source: &'a SafetensorsSource,
    attention_source: &'a dyn TensorSource,
    binding: &'a CheckpointBinding,
    plan: &'a ModelPlan,
}

impl ResidentTextLayer {
    fn load(
        engine: &Engine,
        sources: &ResidentTextSources<'_>,
        plan: &LayerPlan,
        mirror_o_f32: bool,
    ) -> Result<Self, Fail> {
        engine.gpu.ctx.bind_to_thread()?;
        let layer = plan.index as usize;
        let prefix = format!("model.layers.{layer}");
        let input_norm = engine.htod(&read_vector(
            sources.model,
            &format!("{prefix}.input_layernorm.weight"),
            HIDDEN,
        )?)?;
        let MiMoSourceAttention {
            qkv,
            output: source_o_proj,
            sink,
            ..
        } = MiMoSourceAttention::load(
            engine,
            sources.attention_source,
            sources.binding,
            sources.plan,
            layer,
        )?;
        let o_proj = if mirror_o_f32 {
            f32_mirror_of_bf16(engine, source_o_proj)?
        } else {
            source_o_proj
        };
        let post_norm = engine.htod(&read_vector(
            sources.model,
            &format!("{prefix}.post_attention_layernorm.weight"),
            HIDDEN,
        )?)?;
        let dense = if layer == 0 {
            Some(ResidentDense {
                gate: native_fp8_matrix(
                    engine,
                    sources.source,
                    "blk.0.ffn_gate.weight",
                    16384,
                    HIDDEN,
                )?,
                up: native_fp8_matrix(
                    engine,
                    sources.source,
                    "blk.0.ffn_up.weight",
                    16384,
                    HIDDEN,
                )?,
                down: native_fp8_matrix(
                    engine,
                    sources.source,
                    "blk.0.ffn_down.weight",
                    HIDDEN,
                    16384,
                )?,
            })
        } else {
            None
        };
        Ok(Self {
            input_norm,
            qkv,
            sink,
            o_proj,
            post_norm,
            dense,
        })
    }
}

struct ResidentHead {
    norm: CudaSlice<f32>,
    weight: GpuTensor,
}

impl ResidentHead {
    fn load(engine: &Engine, model: &StModel) -> Result<Self, Fail> {
        engine.gpu.ctx.bind_to_thread()?;
        Ok(Self {
            norm: engine.htod(&read_vector(model, "model.norm.weight", HIDDEN)?)?,
            weight: bf16_matrix(engine, model, "lm_head.weight", VOCAB, HIDDEN)?,
        })
    }
}

fn append_kv(
    engine: &Engine,
    slot: &mut Option<KvState>,
    position: usize,
    key: CudaSlice<f32>,
    value: CudaSlice<f32>,
    kv_heads: usize,
) -> Result<usize, Fail> {
    let key_width = kv_heads * QK;
    let value_width = kv_heads * VALUE;
    let device = engine.stream().context().ordinal();
    if position >= MAX_DIAGNOSTIC_TOKENS
        || key.len() != key_width
        || value.len() != value_width
        || key.ordinal() != device
        || value.ordinal() != device
    {
        return Err("MiMo two-token KV append geometry changed".into());
    }
    if position == 0 {
        if slot.is_some() {
            return Err("MiMo KV slot already holds a prior token".into());
        }
        *slot = Some(KvState {
            key,
            value,
            tokens: 1,
        });
        return Ok(0);
    }
    let prior = slot.take().ok_or("MiMo continuation has no cached KV")?;
    if prior.tokens != position
        || prior.key.len() != position * key_width
        || prior.value.len() != position * value_width
        || prior.key.ordinal() != device
        || prior.value.ordinal() != device
    {
        return Err("MiMo continuation KV position or shape differs".into());
    }
    let mut keys = engine.uninit((position + 1) * key_width)?;
    let mut values = engine.uninit((position + 1) * value_width)?;
    engine.dtod_copy_into(&prior.key, &mut keys, 0)?;
    engine.dtod_copy_into(&key, &mut keys, position * key_width)?;
    engine.dtod_copy_into(&prior.value, &mut values, 0)?;
    engine.dtod_copy_into(&value, &mut values, position * value_width)?;
    *slot = Some(KvState {
        key: keys,
        value: values,
        tokens: position + 1,
    });
    Ok(position)
}

fn append_packed_kv(
    engine: &Engine,
    slot: &mut Option<PackedKvState>,
    position: usize,
    key: CudaSlice<u8>,
    value: CudaSlice<u8>,
) -> Result<usize, Fail> {
    const KEY_BYTES: usize = 4 * (QK / 32) * 34;
    const VALUE_BYTES: usize = 4 * (VALUE / 64) * 36;
    let device = engine.stream().context().ordinal();
    if position >= MAX_DIAGNOSTIC_TOKENS
        || key.len() != KEY_BYTES
        || value.len() != VALUE_BYTES
        || key.ordinal() != device
        || value.ordinal() != device
    {
        return Err("MiMo mixed KV append geometry changed".into());
    }
    if position == 0 {
        if slot.is_some() {
            return Err("MiMo mixed KV slot already holds a prior token".into());
        }
        let mut keys = engine.alloc_u8_uninit(MAX_DIAGNOSTIC_TOKENS * KEY_BYTES)?;
        let mut values = engine.alloc_u8_uninit(MAX_DIAGNOSTIC_TOKENS * VALUE_BYTES)?;
        engine
            .stream()
            .memcpy_dtod(&key, &mut keys.slice_mut(..KEY_BYTES))?;
        engine
            .stream()
            .memcpy_dtod(&value, &mut values.slice_mut(..VALUE_BYTES))?;
        *slot = Some(PackedKvState {
            key: keys,
            value: values,
            tokens: 1,
        });
        return Ok(0);
    }
    let mut prior = slot
        .take()
        .ok_or("MiMo mixed continuation has no cached KV")?;
    if prior.tokens != position
        || prior.key.len() != MAX_DIAGNOSTIC_TOKENS * KEY_BYTES
        || prior.value.len() != MAX_DIAGNOSTIC_TOKENS * VALUE_BYTES
        || prior.key.ordinal() != device
        || prior.value.ordinal() != device
    {
        return Err("MiMo mixed KV position or shape differs".into());
    }
    engine.stream().memcpy_dtod(
        &key,
        &mut prior
            .key
            .slice_mut(position * KEY_BYTES..(position + 1) * KEY_BYTES),
    )?;
    engine.stream().memcpy_dtod(
        &value,
        &mut prior
            .value
            .slice_mut(position * VALUE_BYTES..(position + 1) * VALUE_BYTES),
    )?;
    prior.tokens = position + 1;
    *slot = Some(prior);
    Ok(position)
}

fn attention_token(
    engine: &Engine,
    model: &StModel,
    source: &SafetensorsSource,
    plan: &LayerPlan,
    hidden: &CudaSlice<f32>,
    mut cache: AttentionCache<'_>,
    phase: &mut AttentionPhase<'_>,
) -> Result<(CudaSlice<f32>, usize), Fail> {
    let position = phase.turn;
    let layer = plan.index as usize;
    let (attention, global) = checked_attention(plan)?;
    let kv_heads = attention.kv_heads as usize;
    let prefix = format!("model.layers.{layer}");
    let mut phase_start = Instant::now();
    let norm_name = format!("{prefix}.input_layernorm.weight");
    let norm = if let Some(resident) = phase.resident {
        normalized_with_weight(
            engine,
            hidden,
            &resident.input_norm,
            &norm_name,
            &plan.pre_attention_norm,
        )?
    } else {
        normalized(engine, model, hidden, &norm_name, &plan.pre_attention_norm)?
    };
    phase.record(engine, layer, "attention_norm", phase_start)?;
    phase_start = Instant::now();
    let qkv_name = format!("{prefix}.self_attn.qkv_proj.weight");
    let expected_rows = 3072 + kv_heads / 4 * (QK + VALUE);
    let mut projections = Vec::with_capacity(4);
    if let Some(resident) = phase.resident {
        if resident.qkv.len() != 4 {
            return Err(format!("{qkv_name}: resident shard count changed").into());
        }
        for (index, weight) in resident.qkv.iter().enumerate() {
            if weight.out_features() != expected_rows || weight.in_features() != HIDDEN {
                return Err(format!("{qkv_name} resident shard {index}: changed geometry").into());
            }
            projections.push(engine.matmul(weight, &norm, 1)?);
        }
    } else {
        let views = source
            .find_fp8_mimo_qkv_shards(&qkv_name)
            .ok_or_else(|| format!("{qkv_name}: missing four native views"))?;
        if views.len() != 4 {
            return Err(format!("{qkv_name}: wrong source shard count").into());
        }
        for (index, view) in views.iter().enumerate() {
            if view.out_f != expected_rows || view.in_f != HIDDEN {
                return Err(format!("{qkv_name} shard {index}: changed geometry").into());
            }
            let weight = GpuTensor::load_mimo_fp8_qkv_shard(engine, view)?;
            projections.push(engine.matmul(&weight, &norm, 1)?);
        }
    }
    phase.record(engine, layer, "qkv_project", phase_start)?;
    phase_start = Instant::now();
    let mut qkv = engine.mimo_gather_qkv(
        [
            &projections[0],
            &projections[1],
            &projections[2],
            &projections[3],
        ],
        1,
        attention,
    )?;
    drop((projections, norm));
    let position_gpu = engine.htod_i32(&[position as i32])?;
    engine.rope_neox(
        &mut qkv.query,
        &position_gpu,
        QK,
        64,
        64,
        1,
        attention.rope.base,
        1.0,
    )?;
    engine.rope_neox(
        &mut qkv.key,
        &position_gpu,
        QK,
        64,
        kv_heads,
        1,
        attention.rope.base,
        1.0,
    )?;
    phase.record(engine, layer, "qkv_gather_rope", phase_start)?;
    phase_start = Instant::now();
    if !phase.global_only_kv || global {
        if let Some(restored) = fp8_kv_probe(engine, &qkv.key, &qkv.value, phase, layer)? {
            qkv.key = restored.key;
            qkv.value = restored.value;
        }
        if let Some(restored) = q8q5_kv_probe(engine, &qkv.key, &qkv.value, phase, layer)? {
            qkv.key = restored.key;
            qkv.value = restored.value;
        }
        if let Some(restored) = q8q8_kv_probe(engine, &qkv.key, &qkv.value, phase, layer)? {
            qkv.key = restored.key;
            qkv.value = restored.value;
        }
        if let Some(restored) = cpu_kv_pair_probe(engine, &qkv.key, &qkv.value, phase, layer)? {
            qkv.key = restored.key;
            qkv.value = restored.value;
        }
    }
    let streamed_sink = if phase.resident.is_none() && !global {
        Some(engine.htod(&read_vector(
            model,
            &format!("{prefix}.self_attn.attention_sink_bias"),
            64,
        )?)?)
    } else {
        None
    };
    let sink = if let Some(resident) = phase.resident {
        resident.sink.as_ref()
    } else {
        streamed_sink.as_ref()
    };
    let (context, cached_before) = if phase.mixed_gpu && global {
        const KEY_BYTES: usize = 4 * (QK / 32) * 34;
        const DUMMY_VALUE_BYTES: usize = 4 * (VALUE / 32) * 24;
        let encode_start = Instant::now();
        let mut key_codes = engine.alloc_u8_uninit(KEY_BYTES)?;
        let mut dummy_value = engine.alloc_u8_uninit(DUMMY_VALUE_BYTES)?;
        engine.append_kv_quantized(
            &qkv.key,
            &qkv.value,
            &mut key_codes,
            &mut dummy_value,
            0,
            qkv.key.len(),
            qkv.value.len(),
            KEY_BYTES,
            DUMMY_VALUE_BYTES,
            false,
        )?;
        let value_codes = engine.mimo_nvfp4_encode_rows(&qkv.value, VALUE)?;
        phase.record(engine, layer, "mixed_kv_encode", encode_start)?;
        let append_start = Instant::now();
        let cached_before =
            append_packed_kv(engine, cache.packed, position, key_codes, value_codes)?;
        phase.record(engine, layer, "mixed_kv_append", append_start)?;
        let packed = cache
            .packed
            .as_ref()
            .ok_or("MiMo packed KV append lost its state")?;
        let workspace = cache
            .mixed_workspace
            .take()
            .ok_or("MiMo mixed attention has no workspace")?;
        let attention_start = Instant::now();
        let context = engine.mimo_global_q8_nvfp4_decode(
            &qkv.query,
            &packed.key,
            &packed.value,
            position + 1,
            workspace,
        )?;
        phase.record(engine, layer, "mixed_split_attention", attention_start)?;
        (context, cached_before)
    } else {
        let cached_before = append_kv(engine, cache.plain, position, qkv.key, qkv.value, kv_heads)?;
        let plain = cache
            .plain
            .as_ref()
            .ok_or("MiMo KV append lost its state")?;
        (
            engine.mimo_sink_decode(
                &qkv.query,
                &plain.key,
                &plain.value,
                sink,
                position + 1,
                &plan.attention,
            )?,
            cached_before,
        )
    };
    drop((qkv.query, streamed_sink));
    phase.record(engine, layer, "kv_append_attention", phase_start)?;
    phase_start = Instant::now();
    let streamed_output = if phase.resident.is_none() {
        Some(bf16_matrix(
            engine,
            model,
            &format!("{prefix}.self_attn.o_proj.weight"),
            HIDDEN,
            64 * VALUE,
        )?)
    } else {
        None
    };
    let output = if let Some(resident) = phase.resident {
        &resident.o_proj
    } else {
        streamed_output
            .as_ref()
            .ok_or("MiMo streamed attention output weight is missing")?
    };
    let result = engine.matmul(output, &context, 1)?;
    phase.record(engine, layer, "o_project", phase_start)?;
    Ok((result, cached_before))
}

fn native_fp8_matrix(
    engine: &Engine,
    source: &SafetensorsSource,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<GpuTensor, Fail> {
    let view = source
        .find_fp8_native(name)
        .ok_or_else(|| format!("{name}: source FP8 native view missing"))?;
    if view.in_f != cols || view.out_f != rows || view.blk.is_none() {
        return Err(format!("{name}: changed source block-FP8 shape or grid").into());
    }
    let weight = GpuTensor::load_from_source(engine, source, name)?;
    if weight.in_features() != cols
        || weight.out_features() != rows
        || !matches!(
            &weight,
            GpuTensor::Quant {
                qtype: QT_F8_E4M3_BLK,
                ..
            }
        )
    {
        return Err(format!("{name}: did not retain source block-FP8 residency").into());
    }
    Ok(weight)
}

fn dense_mlp_token(
    engine: &Engine,
    source: &SafetensorsSource,
    input: &CudaSlice<f32>,
    plan: &memra_gguf::model_plan::DenseMlpPlan,
) -> Result<CudaSlice<f32>, Fail> {
    if plan.intermediate_size != 16384 || plan.activation != ActivationPlan::Silu {
        return Err("MiMo dense layer 0 MLP contract changed".into());
    }
    let gate = native_fp8_matrix(engine, source, "blk.0.ffn_gate.weight", 16384, HIDDEN)?;
    let gate_out = engine.matmul(&gate, input, 1)?;
    drop(gate);
    let up = native_fp8_matrix(engine, source, "blk.0.ffn_up.weight", 16384, HIDDEN)?;
    let up_out = engine.matmul(&up, input, 1)?;
    drop(up);
    let mut activated = engine.uninit(16384)?;
    engine.silu_mul(&gate_out, &up_out, &mut activated, 16384)?;
    drop((gate_out, up_out));
    let down = native_fp8_matrix(engine, source, "blk.0.ffn_down.weight", HIDDEN, 16384)?;
    engine.matmul(&down, &activated, 1)
}

fn dense_mlp_token_resident(
    engine: &Engine,
    input: &CudaSlice<f32>,
    plan: &memra_gguf::model_plan::DenseMlpPlan,
    weights: &ResidentDense,
) -> Result<CudaSlice<f32>, Fail> {
    if plan.intermediate_size != 16384 || plan.activation != ActivationPlan::Silu {
        return Err("MiMo resident dense layer 0 MLP contract changed".into());
    }
    let gate_out = engine.matmul(&weights.gate, input, 1)?;
    let up_out = engine.matmul(&weights.up, input, 1)?;
    let mut activated = engine.uninit(16384)?;
    engine.silu_mul(&gate_out, &up_out, &mut activated, 16384)?;
    engine.matmul(&weights.down, &activated, 1)
}

fn validate_plan(plan: &ModelPlan) -> Result<(), Fail> {
    if plan.layers.len() != LAYERS
        || plan.hidden_size as usize != HIDDEN
        || plan.vocab_size as usize != VOCAB
        || plan.embedding_scale.to_bits() != 1.0f32.to_bits()
        || !plan.logits.is_empty()
        || plan.output_norm.kind != NormKind::Rms
        || plan.output_norm.weight_transform != WeightTransform::Identity
        || !plan.partition_boundaries.contains(&STAGE_CUT)
    {
        return Err("MiMo text plan geometry changed".into());
    }
    for (index, layer) in plan.layers.iter().enumerate() {
        if layer.index as usize != index
            || layer.pre_attention_norm.kind != NormKind::Rms
            || layer.pre_attention_norm.weight_transform != WeightTransform::Identity
            || layer.pre_mlp_norm.kind != NormKind::Rms
            || layer.pre_mlp_norm.weight_transform != WeightTransform::Identity
        {
            return Err(format!("MiMo layer {index}: norm or index changed").into());
        }
        checked_attention(layer)?;
        match (&layer.mlp, index) {
            (MlpPlan::Dense(dense), 0)
                if dense.intermediate_size == 16384 && dense.activation == ActivationPlan::Silu => {
            }
            (MlpPlan::Moe(moe), 1..)
                if moe.expert_count == 256
                    && moe.experts_per_token == 8
                    && moe.expert_intermediate_size == 2048
                    && moe.activation == ActivationPlan::Silu
                    && moe.shared.is_none() => {}
            _ => return Err(format!("MiMo layer {index}: MLP plan changed").into()),
        }
    }
    Ok(())
}

fn device_memory(engine: &Engine) -> Result<(usize, usize), Fail> {
    engine.gpu.ctx.bind_to_thread()?;
    Ok(engine.stream().context().mem_get_info()?)
}

fn run() -> Result<(), Fail> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 5 || args.len() > 21 {
        return Err(
            "usage: mimo_source_gpu_token <source_dir> <gpu0> <gpu1> <token_id> <report.tsv> [--continue-one | --tokens=N] [--prompt-ids-file=PATH] [--resident-moe] [--resident-text] [--grouped-moe] [--mirror-o-f32] [--profile-phases] [--capacity-probe=N] [--capacity-mixed-kv] [--workspace-mib=N] [--kv-global-only] [--kv-fp8-probe | --kv-fp8-replay | --kv-q8q5-probe | --kv-q8q5-replay | --kv-q8q8-probe | --kv-q8q8-replay | --kv-cpu-pair=K-V | --kv-mixed-gpu] [--mixed-grouped-attn | --mixed-deep-attn | --mixed-dp4a-attn | --mixed-native-vscale]"
                .into(),
        );
    }
    let mut continue_one = false;
    let mut requested_turns: Option<usize> = None;
    let mut prompt_ids_path: Option<&str> = None;
    let mut capacity_sessions: Option<usize> = None;
    let mut capacity_mixed_kv = false;
    let mut workspace_mib: Option<usize> = None;
    let mut resident_moe = false;
    let mut resident_text = false;
    let mut grouped_moe = false;
    let mut mirror_o_f32 = false;
    let mut profile_phases = false;
    let mut fp8_probe = false;
    let mut fp8_replay = false;
    let mut q8q5_probe = false;
    let mut q8q5_replay = false;
    let mut q8q8_probe = false;
    let mut q8q8_replay = false;
    let mut global_only_kv = false;
    let mut cpu_pair: Option<CpuKvPair> = None;
    let mut mixed_gpu = false;
    let mut mixed_grouped_attn = false;
    let mut mixed_deep_attn = false;
    let mut mixed_dp4a_attn = false;
    let mut mixed_native_vscale = false;
    for option in args.iter().skip(5) {
        match option.as_str() {
            "--continue-one" if !continue_one => continue_one = true,
            "--resident-moe" if !resident_moe => resident_moe = true,
            "--resident-text" if !resident_text => resident_text = true,
            "--grouped-moe" if !grouped_moe => grouped_moe = true,
            "--mirror-o-f32" if !mirror_o_f32 => mirror_o_f32 = true,
            "--profile-phases" if !profile_phases => profile_phases = true,
            "--kv-fp8-probe" if !fp8_probe => fp8_probe = true,
            "--kv-fp8-replay" if !fp8_replay => fp8_replay = true,
            "--kv-q8q5-probe" if !q8q5_probe => q8q5_probe = true,
            "--kv-q8q5-replay" if !q8q5_replay => q8q5_replay = true,
            "--kv-q8q8-probe" if !q8q8_probe => q8q8_probe = true,
            "--kv-q8q8-replay" if !q8q8_replay => q8q8_replay = true,
            "--kv-global-only" if !global_only_kv => global_only_kv = true,
            "--kv-mixed-gpu" if !mixed_gpu => mixed_gpu = true,
            "--mixed-grouped-attn" if !mixed_grouped_attn => mixed_grouped_attn = true,
            "--mixed-deep-attn" if !mixed_deep_attn => mixed_deep_attn = true,
            "--mixed-dp4a-attn" if !mixed_dp4a_attn => mixed_dp4a_attn = true,
            "--mixed-native-vscale" if !mixed_native_vscale => mixed_native_vscale = true,
            "--capacity-mixed-kv" if !capacity_mixed_kv => capacity_mixed_kv = true,
            _ if option.starts_with("--kv-cpu-pair=") && cpu_pair.is_none() => {
                cpu_pair = Some(
                    cpu_kv_pair(&option["--kv-cpu-pair=".len()..])
                        .ok_or("unsupported MiMo CPU KV pair")?,
                );
            }
            _ if option.starts_with("--tokens=") && requested_turns.is_none() => {
                requested_turns = Some(option["--tokens=".len()..].parse()?);
            }
            _ if option.starts_with("--prompt-ids-file=") && prompt_ids_path.is_none() => {
                prompt_ids_path = Some(&option["--prompt-ids-file=".len()..]);
            }
            _ if option.starts_with("--capacity-probe=") && capacity_sessions.is_none() => {
                capacity_sessions = Some(option["--capacity-probe=".len()..].parse()?);
            }
            _ if option.starts_with("--workspace-mib=") && workspace_mib.is_none() => {
                workspace_mib = Some(option["--workspace-mib=".len()..].parse()?);
            }
            _ => return Err(format!("unknown or repeated MiMo token option: {option}").into()),
        }
    }
    if continue_one && (requested_turns.is_some() || prompt_ids_path.is_some()) {
        return Err("--continue-one cannot combine with --tokens or prompt IDs".into());
    }
    if fp8_replay {
        fp8_probe = true;
    }
    if q8q5_replay {
        q8q5_probe = true;
    }
    if q8q8_replay {
        q8q8_probe = true;
    }
    if cpu_pair.is_some() || mixed_gpu {
        global_only_kv = true;
    }
    if u8::from(fp8_probe) + u8::from(q8q5_probe) + u8::from(q8q8_probe) + u8::from(mixed_gpu) > 1 {
        return Err("MiMo KV probe must select one storage format".into());
    }
    if cpu_pair.is_some() && (fp8_probe || q8q5_probe || q8q8_probe || mixed_gpu) {
        return Err("MiMo CPU KV pair conflicts with a GPU KV format probe".into());
    }
    if global_only_kv && !(fp8_probe || q8q5_probe || q8q8_probe || cpu_pair.is_some() || mixed_gpu)
    {
        return Err("--kv-global-only requires one KV storage probe".into());
    }
    let turns = requested_turns.unwrap_or(if continue_one { 2 } else { 1 });
    if !(1..=MAX_DIAGNOSTIC_TOKENS).contains(&turns) {
        return Err("MiMo diagnostic token count is outside 1..=256".into());
    }
    if let Some(sessions) = capacity_sessions {
        if !(1..=2).contains(&sessions)
            || requested_turns.is_some()
            || continue_one
            || profile_phases
            || fp8_probe
            || q8q5_probe
            || q8q8_probe
            || cpu_pair.is_some()
            || mixed_gpu
            || global_only_kv
            || prompt_ids_path.is_some()
            || !resident_text
            || workspace_mib.is_some_and(|mib| mib > 8192)
        {
            return Err("MiMo capacity probe requires 1-2 sessions, resident text, and no token/profile option".into());
        }
    } else if workspace_mib.is_some() || capacity_mixed_kv {
        return Err("--workspace-mib and --capacity-mixed-kv require --capacity-probe".into());
    }
    if mixed_grouped_attn && !(mixed_gpu || capacity_mixed_kv) {
        return Err("--mixed-grouped-attn requires mixed GPU KV or mixed capacity".into());
    }
    if mixed_deep_attn && !(mixed_gpu || capacity_mixed_kv) {
        return Err("--mixed-deep-attn requires mixed GPU KV or mixed capacity".into());
    }
    if mixed_dp4a_attn && !(mixed_gpu || capacity_mixed_kv) {
        return Err("--mixed-dp4a-attn requires mixed GPU KV or mixed capacity".into());
    }
    if mixed_native_vscale && !(mixed_gpu || capacity_mixed_kv) {
        return Err("--mixed-native-vscale requires mixed GPU KV or mixed capacity".into());
    }
    if u8::from(mixed_grouped_attn)
        + u8::from(mixed_deep_attn)
        + u8::from(mixed_dp4a_attn)
        + u8::from(mixed_native_vscale)
        > 1
    {
        return Err("MiMo mixed attention must select one schedule".into());
    }
    if resident_text {
        resident_moe = true;
    }
    if grouped_moe && !resident_text {
        return Err("MiMo grouped MoE diagnostic requires --resident-text".into());
    }
    if (fp8_probe || q8q5_probe || q8q8_probe || cpu_pair.is_some() || mixed_gpu) && !resident_text
    {
        return Err("MiMo KV storage probe requires --resident-text".into());
    }
    if mirror_o_f32 && !resident_text {
        return Err("MiMo f32 attention output mirror requires --resident-text".into());
    }
    let direct_bf16 = match std::env::var("MEMRA_BF16_MMV") {
        Ok(value) if value == "1" => true,
        Ok(value) if value == "0" => false,
        Err(std::env::VarError::NotPresent) => false,
        _ => return Err("MiMo diagnostic requires MEMRA_BF16_MMV to be 0, 1, or unset".into()),
    };
    if direct_bf16 && mirror_o_f32 {
        return Err("MiMo f32 attention output mirror conflicts with direct BF16 matvec".into());
    }
    if grouped_moe && direct_bf16 {
        return Err("MiMo grouped MoE diagnostic requires f32 output accumulation".into());
    }
    if (fp8_replay || q8q5_replay || q8q8_replay || cpu_pair.is_some() || mixed_gpu) && direct_bf16
    {
        return Err("MiMo KV replay cannot combine with direct BF16 matvec".into());
    }
    let dir = Path::new(&args[0]);
    let gpu0: usize = args[1].parse()?;
    let gpu1: usize = args[2].parse()?;
    let token: usize = args[3].parse()?;
    let (prompt_tokens, prompt_sha256) = if let Some(path) = prompt_ids_path {
        let bytes = std::fs::read(path)?;
        let digest = format!("{:x}", Sha256::digest(&bytes));
        let text = std::str::from_utf8(&bytes)?;
        let ids: Vec<usize> = text
            .lines()
            .map(|line| {
                if line.is_empty() || line.trim() != line {
                    Err("MiMo prompt ID file has an empty or padded line".into())
                } else {
                    line.parse::<usize>().map_err(Into::into)
                }
            })
            .collect::<Result<_, Fail>>()?;
        if ids.is_empty()
            || ids.len() > turns
            || ids[0] != token
            || ids.iter().any(|&id| id >= VOCAB)
        {
            return Err(
                "MiMo prompt IDs must be nonempty, fit turns/vocab, and start at token_id".into(),
            );
        }
        (ids, digest)
    } else {
        (Vec::new(), String::from("none"))
    };
    let report_path = Path::new(&args[4]);
    let staged_path = report_path.with_extension("tsv.writing");
    if gpu0 == gpu1 || report_path.exists() || staged_path.exists() {
        return Err("MiMo stage devices must differ and report paths must not exist".into());
    }
    let started = Instant::now();
    let preflight_start = Instant::now();
    let config_bytes = std::fs::read(dir.join("config.json"))?;
    let digest = format!("{:x}", Sha256::digest(&config_bytes));
    if digest != CONFIG_SHA256 {
        return Err("MiMo source config differs from pinned revision".into());
    }
    let config_text = String::from_utf8(config_bytes)?;
    let config = ModelConfig::from_hf(&HfConfig::try_parse(&config_text)?);
    let pack = model_packs::by_alias("mimo_v2_source")
        .ok_or("MiMo source inspection profile unavailable")?;
    let plan = pack.compile_plan(&config)?;
    validate_plan(&plan)?;
    if !memra_engine::fp8_ffi::st_e4m3_blk_enabled() {
        return Err("MiMo source token requires native block-FP8 residency".into());
    }
    let model = StModel::open(dir)?;
    let pinned = PinnedMiMoSource::bind(&model, &config, &plan)?;
    if pinned.semantic_tensors() != 36922 {
        return Err("MiMo source semantic binding count changed".into());
    }
    let source = SafetensorsSource::open(dir)?;
    let (_, bound_plan, binding) = model_packs::mimo_v2::bind_pinned_text_source(&source)?;
    if bound_plan != plan || binding.bound.tensors.len() != 36922 {
        return Err("MiMo source semantic binding differs from pinned diagnostic plan".into());
    }
    for layer in &plan.layers {
        let index = layer.index as usize;
        let (attention, _) = checked_attention(layer)?;
        let name = format!("model.layers.{index}.self_attn.qkv_proj.weight");
        let views = source
            .find_fp8_mimo_qkv_shards(&name)
            .ok_or_else(|| format!("{name}: missing native QKV shards"))?;
        let expected_rows = 3072 + (attention.kv_heads as usize / 4) * (QK + VALUE);
        if views.len() != 4
            || views
                .iter()
                .any(|view| view.out_f != expected_rows || view.in_f != HIDDEN)
        {
            return Err(format!("{name}: native QKV shard geometry changed").into());
        }
    }
    for (name, rows, cols) in [
        ("blk.0.ffn_gate.weight", 16384, HIDDEN),
        ("blk.0.ffn_up.weight", 16384, HIDDEN),
        ("blk.0.ffn_down.weight", HIDDEN, 16384),
    ] {
        let view = source
            .find_fp8_native(name)
            .ok_or_else(|| format!("{name}: native block-FP8 view missing"))?;
        if view.out_f != rows || view.in_f != cols || view.blk.is_none() {
            return Err(format!("{name}: native block-FP8 geometry changed").into());
        }
    }
    embedding_row(&model, token)?;
    let preflight_ms = preflight_start.elapsed().as_secs_f64() * 1000.0;
    let engines = [Engine::new(gpu0)?, Engine::new(gpu1)?];
    let mut resident_layers: Vec<Option<ResidentMiMoMoeLayer>> =
        std::iter::repeat_with(|| None).take(LAYERS).collect();
    let mut grouped_layers: Vec<Option<GroupedMiMoMoeLayer>> =
        std::iter::repeat_with(|| None).take(LAYERS).collect();
    let mut resident_bytes = [0usize; 2];
    let mut resident_load_ms = 0.0f64;
    if resident_moe {
        let load_start = Instant::now();
        for (index, layer) in plan.layers.iter().enumerate().skip(1) {
            let stage = usize::from(index >= STAGE_CUT);
            let engine = &engines[stage];
            engine.gpu.ctx.bind_to_thread()?;
            let MlpPlan::Moe(moe) = &layer.mlp else {
                return Err(format!("MiMo resident layer {index} has no MoE plan").into());
            };
            let before = Instant::now();
            if grouped_moe {
                let grouped = GroupedMiMoMoeLayer::load(engine, &pinned, index, moe)?;
                resident_bytes[stage] += grouped.resident_bytes();
                grouped_layers[index] = Some(grouped);
            } else {
                let resident = ResidentMiMoMoeLayer::load(engine, &pinned, index, moe)?;
                resident_bytes[stage] += resident.resident_bytes();
                resident_layers[index] = Some(resident);
            }
            eprintln!(
                "MiMo resident layer {index} stage {stage} loaded in {:.3}s; cumulative stage bytes {}",
                before.elapsed().as_secs_f64(),
                resident_bytes[stage]
            );
        }
        resident_load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    }
    let mut resident_text_layers: Vec<Option<ResidentTextLayer>> =
        std::iter::repeat_with(|| None).take(LAYERS).collect();
    let mut resident_head: Option<ResidentHead> = None;
    let mut resident_text_load_ms = 0.0f64;
    if resident_text {
        let attention_source = RecordingSource::new(&source);
        let text_sources = ResidentTextSources {
            model: &model,
            source: &source,
            attention_source: &attention_source,
            binding: &binding,
            plan: &plan,
        };
        let load_start = Instant::now();
        for (index, layer) in plan.layers.iter().enumerate() {
            let stage = usize::from(index >= STAGE_CUT);
            let engine = &engines[stage];
            let before = Instant::now();
            resident_text_layers[index] = Some(ResidentTextLayer::load(
                engine,
                &text_sources,
                layer,
                mirror_o_f32,
            )?);
            eprintln!(
                "MiMo text layer {index} stage {stage} resident in {:.3}s",
                before.elapsed().as_secs_f64()
            );
        }
        let missing = binding.audit_consumption(&attention_source.requested(), &config, |id, _| {
            !matches!(
                id,
                memra_gguf::tensor_contract::TensorId::Layer {
                    tensor: memra_gguf::tensor_contract::LayerTensor::FusedQkv
                        | memra_gguf::tensor_contract::LayerTensor::AttentionOutput
                        | memra_gguf::tensor_contract::LayerTensor::AttentionSink,
                    ..
                }
            )
        });
        if !missing.is_empty() {
            return Err(
                format!("MiMo attention source left bound tensors unread: {missing:?}").into(),
            );
        }
        resident_head = Some(ResidentHead::load(&engines[1], &model)?);
        resident_text_load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    }
    if let Some(sessions) = capacity_sessions {
        let workspace_bytes = workspace_mib.unwrap_or(0) * 1024 * 1024;
        let attention_scratch_bytes = if capacity_mixed_kv {
            let tiles: usize = if mixed_deep_attn || mixed_dp4a_attn || mixed_native_vscale {
                2048
            } else if mixed_grouped_attn {
                16384
            } else {
                4096
            };
            64 * (tiles + tiles.div_ceil(128) + 1) * (VALUE + 2) * 4
        } else {
            0
        };
        let before = [device_memory(&engines[0])?, device_memory(&engines[1])?];
        let mut held = [Vec::<CudaSlice<u8>>::new(), Vec::<CudaSlice<u8>>::new()];
        let mut kv_bytes = [0usize; 2];
        let mut plane_count = [0usize; 2];
        let mut workspace_allocated = [false; 2];
        let mut attention_scratch_allocated = [false; 2];
        let mut failure: Option<String> = None;
        'layers: for (index, layer) in plan.layers.iter().enumerate() {
            let stage = usize::from(index >= STAGE_CUT);
            let engine = &engines[stage];
            engine.gpu.ctx.bind_to_thread()?;
            let (attention, global) = checked_attention(layer)?;
            let stored_tokens = if global { 1_048_576 } else { 128 };
            let kv_heads = attention.kv_heads as usize;
            for session in 0..sessions {
                for (plane, width) in [("key", kv_heads * QK), ("value", kv_heads * VALUE)] {
                    let bytes = match (capacity_mixed_kv, global, plane) {
                        (true, true, "key") => stored_tokens * kv_heads * (QK / 32) * 34,
                        (true, true, "value") => stored_tokens * kv_heads * (VALUE / 64) * 36,
                        (true, false, _) => stored_tokens * width * 4,
                        _ => stored_tokens * width,
                    };
                    match engine.alloc_u8(bytes) {
                        Ok(buffer) => {
                            held[stage].push(buffer);
                            kv_bytes[stage] += bytes;
                            plane_count[stage] += 1;
                        }
                        Err(error) => {
                            failure = Some(format!(
                                "stage {stage} layer {index} session {session} {plane} allocation of {bytes} bytes: {error}"
                            ));
                            break 'layers;
                        }
                    }
                }
            }
        }
        if failure.is_none() && attention_scratch_bytes > 0 {
            for stage in 0..2 {
                let engine = &engines[stage];
                engine.gpu.ctx.bind_to_thread()?;
                match engine.alloc_u8(attention_scratch_bytes) {
                    Ok(buffer) => {
                        held[stage].push(buffer);
                        attention_scratch_allocated[stage] = true;
                    }
                    Err(error) => {
                        failure = Some(format!(
                            "stage {stage} split attention scratch allocation of {attention_scratch_bytes} bytes: {error}"
                        ));
                        break;
                    }
                }
            }
        }
        if failure.is_none() && workspace_bytes > 0 {
            for stage in 0..2 {
                let engine = &engines[stage];
                engine.gpu.ctx.bind_to_thread()?;
                match engine.alloc_u8(workspace_bytes) {
                    Ok(buffer) => {
                        held[stage].push(buffer);
                        workspace_allocated[stage] = true;
                    }
                    Err(error) => {
                        failure = Some(format!(
                            "stage {stage} workspace allocation of {workspace_bytes} bytes: {error}"
                        ));
                        break;
                    }
                }
            }
        }
        let mut after = [None, None];
        for stage in 0..2 {
            if let Err(error) = engines[stage].stream().synchronize() {
                failure.get_or_insert_with(|| format!("stage {stage} synchronization: {error}"));
            }
            match device_memory(&engines[stage]) {
                Ok(info) => after[stage] = Some(info),
                Err(error) => {
                    failure.get_or_insert_with(|| format!("stage {stage} memory query: {error}"));
                }
            }
        }
        let mut report = String::from("format\tmemra-mimo-resident-kv-capacity-v1\n");
        writeln!(report, "model\tXiaomiMiMo/MiMo-V2.6-Flash-RL@{REVISION}")?;
        writeln!(report, "config_sha256\t{CONFIG_SHA256}")?;
        writeln!(
            report,
            "source_header_digest\t5ebbdd27e45716b805fc2bdf115c8345b4bfc03c6012b860f76b6b222c758aee"
        )?;
        writeln!(
            report,
            "scope\t{}",
            if capacity_mixed_kv {
                "global_q8_0_k_nvfp4_v_sliding_f32_physical_allocation"
            } else {
                "raw_one_byte_kv_planes_no_scales_no_attention"
            }
        )?;
        writeln!(report, "sessions\t{sessions}")?;
        writeln!(report, "context_tokens_per_session\t1048576")?;
        writeln!(report, "sliding_window_tokens\t128")?;
        writeln!(report, "stage_cut_before_layer\t{STAGE_CUT}")?;
        writeln!(report, "grouped_moe\t{grouped_moe}")?;
        writeln!(report, "o_proj_f32_mirror\t{mirror_o_f32}")?;
        writeln!(
            report,
            "workspace_reserve_bytes_per_card\t{workspace_bytes}"
        )?;
        writeln!(
            report,
            "split_attention_scratch_bytes_per_card\t{attention_scratch_bytes}"
        )?;
        writeln!(report, "mixed_grouped_attention\t{mixed_grouped_attn}")?;
        writeln!(report, "mixed_deep_attention\t{mixed_deep_attn}")?;
        writeln!(report, "mixed_dp4a_attention\t{mixed_dp4a_attn}")?;
        writeln!(report, "mixed_native_vscale\t{mixed_native_vscale}")?;
        writeln!(report, "resident_moe_load_ms\t{resident_load_ms:.3}")?;
        writeln!(report, "resident_text_load_ms\t{resident_text_load_ms:.3}")?;
        for stage in 0..2 {
            writeln!(report, "device_ordinal\t{stage}\t{}", [gpu0, gpu1][stage])?;
            writeln!(report, "memory_total_bytes\t{stage}\t{}", before[stage].1)?;
            writeln!(
                report,
                "memory_free_before_bytes\t{stage}\t{}",
                before[stage].0
            )?;
            writeln!(report, "kv_plane_count\t{stage}\t{}", plane_count[stage])?;
            writeln!(report, "kv_bytes_allocated\t{stage}\t{}", kv_bytes[stage])?;
            writeln!(
                report,
                "workspace_allocated\t{stage}\t{}",
                workspace_allocated[stage]
            )?;
            writeln!(
                report,
                "split_attention_scratch_allocated\t{stage}\t{}",
                attention_scratch_allocated[stage]
            )?;
            if let Some((free, _)) = after[stage] {
                writeln!(report, "memory_free_after_bytes\t{stage}\t{free}")?;
            }
        }
        writeln!(report, "passed\t{}", failure.is_none())?;
        writeln!(report, "failure\t{}", failure.as_deref().unwrap_or("none"))?;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&staged_path)?;
        file.write_all(report.as_bytes())?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(staged_path, report_path)?;
        drop(held);
        return if let Some(message) = failure {
            Err(message.into())
        } else {
            Ok(())
        };
    }
    let mut kv: Vec<Option<KvState>> = std::iter::repeat_with(|| None).take(LAYERS).collect();
    let mut packed_kv: Vec<Option<PackedKvState>> =
        std::iter::repeat_with(|| None).take(LAYERS).collect();
    let mut mixed_workspaces = if mixed_gpu {
        Some(if mixed_native_vscale {
            [
                MiMoMixedAttentionWorkspace::new_dp4a_native_vscale(&engines[0], turns)?,
                MiMoMixedAttentionWorkspace::new_dp4a_native_vscale(&engines[1], turns)?,
            ]
        } else if mixed_dp4a_attn {
            [
                MiMoMixedAttentionWorkspace::new_dp4a(&engines[0], turns)?,
                MiMoMixedAttentionWorkspace::new_dp4a(&engines[1], turns)?,
            ]
        } else if mixed_deep_attn {
            [
                MiMoMixedAttentionWorkspace::new_deep(&engines[0], turns)?,
                MiMoMixedAttentionWorkspace::new_deep(&engines[1], turns)?,
            ]
        } else if mixed_grouped_attn {
            [
                MiMoMixedAttentionWorkspace::new_grouped(&engines[0], turns)?,
                MiMoMixedAttentionWorkspace::new_grouped(&engines[1], turns)?,
            ]
        } else {
            [
                MiMoMixedAttentionWorkspace::new(&engines[0], turns)?,
                MiMoMixedAttentionWorkspace::new(&engines[1], turns)?,
            ]
        })
    } else {
        None
    };
    let mut report = if turns > 1 {
        String::from("format\tmemra-mimo-source-gpu-token-v2\n")
    } else {
        String::from("format\tmemra-mimo-source-gpu-token-v1\n")
    };
    let cpu_pair_numeric = cpu_pair.map(|pair| {
        format!(
            "memra_mimo_source_global_{}_cpu_roundtrip_candidate",
            pair.label
        )
    });
    let cpu_pair_format =
        cpu_pair.map(|pair| format!("global_{}_sliding_f32_cpu_component", pair.label));
    writeln!(report, "model\tXiaomiMiMo/MiMo-V2.6-Flash-RL@{REVISION}")?;
    writeln!(report, "payload_verification\texternal_hf_verify_required")?;
    writeln!(report, "config_sha256\t{CONFIG_SHA256}")?;
    writeln!(
        report,
        "source_header_digest\t5ebbdd27e45716b805fc2bdf115c8345b4bfc03c6012b860f76b6b222c758aee"
    )?;
    writeln!(
        report,
        "numeric_class\t{}",
        if mixed_gpu && mixed_native_vscale {
            "memra_mimo_source_global_q8_0_k_nvfp4_v_q8_query_dp4a_native_vscale_candidate"
        } else if mixed_gpu && mixed_dp4a_attn {
            "memra_mimo_source_global_q8_0_k_nvfp4_v_q8_query_dp4a_deep_candidate"
        } else if mixed_gpu && mixed_deep_attn {
            "memra_mimo_source_global_q8_0_k_nvfp4_v_native_deep_split_candidate"
        } else if mixed_gpu && mixed_grouped_attn {
            "memra_mimo_source_global_q8_0_k_nvfp4_v_native_grouped_split_candidate"
        } else if mixed_gpu {
            "memra_mimo_source_global_q8_0_k_nvfp4_v_native_split_candidate"
        } else if let Some(name) = cpu_pair_numeric.as_deref() {
            name
        } else if q8q8_replay && global_only_kv {
            "memra_mimo_source_global_q8_0_k_q8_0_v_sliding_f32_candidate"
        } else if q8q5_replay && global_only_kv {
            "memra_mimo_source_global_q8_0_k_q5_1_v_sliding_f32_candidate"
        } else if fp8_replay && global_only_kv {
            "memra_mimo_source_global_e4m3_kv_sliding_f32_candidate"
        } else if q8q8_replay {
            "memra_mimo_source_q8_0_k_q8_0_v_roundtrip_f32_attention_candidate"
        } else if q8q5_replay {
            "memra_mimo_source_q8_0_k_q5_1_v_roundtrip_f32_attention_candidate"
        } else if fp8_replay {
            "memra_mimo_source_e4m3_kv_roundtrip_f32_attention_candidate"
        } else if direct_bf16 {
            "memra_block_fp8_q8_1_act_mxfp4_pow2_e4m3_moe_act_bf16_direct_f32acc"
        } else if grouped_moe && mirror_o_f32 {
            "memra_block_fp8_q8_1_act_mxfp4_pow2_e4m3_grouped_moe_o_f32_candidate"
        } else if grouped_moe {
            "memra_block_fp8_q8_1_act_mxfp4_pow2_e4m3_grouped_moe_candidate"
        } else if mirror_o_f32 {
            "memra_block_fp8_q8_1_act_mxfp4_pow2_e4m3_moe_act_f32_o_mirror_candidate"
        } else {
            "memra_block_fp8_q8_1_act_and_mxfp4_pow2_e4m3_moe_act"
        }
    )?;
    writeln!(report, "direct_bf16_matvec\t{direct_bf16}")?;
    writeln!(report, "grouped_moe\t{grouped_moe}")?;
    writeln!(report, "o_proj_f32_mirror\t{mirror_o_f32}")?;
    writeln!(report, "kv_fp8_probe\t{fp8_probe}")?;
    writeln!(report, "kv_fp8_replay\t{fp8_replay}")?;
    writeln!(report, "kv_q8q5_probe\t{q8q5_probe}")?;
    writeln!(report, "kv_q8q5_replay\t{q8q5_replay}")?;
    writeln!(report, "kv_q8q8_probe\t{q8q8_probe}")?;
    writeln!(report, "kv_q8q8_replay\t{q8q8_replay}")?;
    writeln!(report, "kv_mixed_gpu\t{mixed_gpu}")?;
    writeln!(report, "mixed_grouped_attention\t{mixed_grouped_attn}")?;
    writeln!(report, "mixed_deep_attention\t{mixed_deep_attn}")?;
    writeln!(report, "mixed_dp4a_attention\t{mixed_dp4a_attn}")?;
    writeln!(report, "mixed_native_vscale\t{mixed_native_vscale}")?;
    writeln!(report, "kv_global_only\t{global_only_kv}")?;
    writeln!(
        report,
        "kv_cpu_pair\t{}",
        cpu_pair.map_or("none", |pair| pair.label)
    )?;
    writeln!(report, "stage_cut_before_layer\t{STAGE_CUT}")?;
    writeln!(report, "stage_transfer\thost_bounce")?;
    writeln!(
        report,
        "weight_residency\t{}",
        if grouped_moe && mirror_o_f32 {
            "source_grouped_moe_qkv_o_f32_norms_head_resident_embedding_row_streamed"
        } else if grouped_moe {
            "source_grouped_moe_qkv_o_bf16_norms_head_resident_embedding_row_streamed"
        } else if mirror_o_f32 {
            "source_moe_qkv_o_f32_norms_head_resident_embedding_row_streamed"
        } else if resident_text {
            "source_moe_projections_norms_head_resident_embedding_row_streamed"
        } else if resident_moe {
            "source_mxfp4_moe_resident"
        } else {
            "layer_streamed"
        }
    )?;
    writeln!(report, "resident_moe_bytes_stage0\t{}", resident_bytes[0])?;
    writeln!(report, "resident_moe_bytes_stage1\t{}", resident_bytes[1])?;
    writeln!(report, "resident_moe_load_ms\t{resident_load_ms:.3}")?;
    writeln!(report, "resident_text_load_ms\t{resident_text_load_ms:.3}")?;
    writeln!(report, "source_preflight_ms\t{preflight_ms:.3}")?;
    writeln!(report, "phase_timing\t{profile_phases}")?;
    writeln!(
        report,
        "kv_format\t{}",
        if mixed_gpu && mixed_native_vscale {
            "global_q8_0_k_nvfp4_v_native_dp4a_fp8_scale_split_sliding_f32"
        } else if mixed_gpu && mixed_dp4a_attn {
            "global_q8_0_k_nvfp4_v_native_dp4a_split_sliding_f32"
        } else if mixed_gpu && mixed_deep_attn {
            "global_q8_0_k_nvfp4_v_native_deep_split_sliding_f32"
        } else if mixed_gpu && mixed_grouped_attn {
            "global_q8_0_k_nvfp4_v_native_grouped_split_sliding_f32"
        } else if mixed_gpu {
            "global_q8_0_k_nvfp4_v_native_split_sliding_f32"
        } else if let Some(name) = cpu_pair_format.as_deref() {
            name
        } else if q8q8_replay && global_only_kv {
            "global_q8_0_k_q8_0_v_sliding_f32_component"
        } else if q8q5_replay && global_only_kv {
            "global_q8_0_k_q5_1_v_sliding_f32_component"
        } else if fp8_replay && global_only_kv {
            "global_e4m3_kv_sliding_f32_component"
        } else if q8q8_replay {
            "q8_0_k_q8_0_v_roundtripped_f32_contiguous_component"
        } else if q8q5_replay {
            "q8_0_k_q5_1_v_roundtripped_f32_contiguous_component"
        } else if fp8_replay {
            "e4m3_roundtripped_f32_contiguous_component"
        } else {
            "f32_contiguous_component"
        }
    )?;
    writeln!(
        report,
        "kv_append\t{}",
        if mixed_gpu {
            "global_packed_preallocated_append_sliding_f32_copy"
        } else {
            "device_copy_of_prior_plus_current"
        }
    )?;
    writeln!(report, "kv_sequence_length\t{turns}")?;
    writeln!(report, "prompt_token_count\t{}", prompt_tokens.len())?;
    writeln!(report, "prompt_ids_sha256\t{prompt_sha256}")?;
    writeln!(
        report,
        "completion_start_turn\t{}",
        prompt_tokens.len().saturating_sub(1)
    )?;
    writeln!(report, "gpu0_ordinal\t{gpu0}")?;
    writeln!(report, "gpu1_ordinal\t{gpu1}")?;
    writeln!(report, "token_id\t{token}")?;
    writeln!(
        report,
        "generated_continuation\t{}",
        turns > prompt_tokens.len().max(1)
    )?;
    let mut current_token = token;
    let mut last_logits: Option<Vec<f32>> = None;
    let mut last_argmax: Option<usize> = None;
    for turn in 0..turns {
        let turn_start = Instant::now();
        if let Some(&forced) = prompt_tokens.get(turn) {
            current_token = forced;
        }
        if turn > 0
            && (0..LAYERS).any(|index| {
                if mixed_gpu && GLOBAL_LAYERS.contains(&index) {
                    packed_kv[index]
                        .as_ref()
                        .is_none_or(|cached| cached.tokens != turn)
                } else {
                    kv[index]
                        .as_ref()
                        .is_none_or(|cached| cached.tokens != turn)
                }
            })
        {
            return Err("MiMo continuation did not retain every layer's native KV".into());
        }
        engines[0].gpu.ctx.bind_to_thread()?;
        let initial = embedding_row(&model, current_token)?;
        let mut hidden = engines[0].htod(&initial)?;
        writeln!(report, "processed_token\t{turn}\t{current_token}")?;
        for (index, layer) in plan.layers.iter().enumerate() {
            let stage = usize::from(index >= STAGE_CUT);
            if index == STAGE_CUT {
                let transfer_start = Instant::now();
                let values = engines[0].dtoh(&hidden)?;
                if values.len() != HIDDEN || values.iter().any(|value| !value.is_finite()) {
                    return Err("MiMo stage transfer carried invalid hidden values".into());
                }
                engines[1].gpu.ctx.bind_to_thread()?;
                hidden = engines[1].htod(&values)?;
                record_phase(
                    &mut report,
                    &engines[1],
                    profile_phases,
                    turn,
                    index,
                    "stage_transfer",
                    transfer_start,
                )?;
            }
            let engine = &engines[stage];
            engine.gpu.ctx.bind_to_thread()?;
            let before = Instant::now();
            let text_layer = if resident_text {
                Some(
                    resident_text_layers[index]
                        .as_ref()
                        .ok_or("MiMo text layer was not resident")?,
                )
            } else {
                None
            };
            let (attention, cached_before) = {
                let mut phase = AttentionPhase {
                    report: &mut report,
                    enabled: profile_phases,
                    turn,
                    resident: text_layer,
                    fp8_probe,
                    fp8_replay,
                    q8q5_probe,
                    q8q5_replay,
                    q8q8_probe,
                    q8q8_replay,
                    global_only_kv,
                    cpu_pair,
                    mixed_gpu,
                };
                attention_token(
                    engine,
                    &model,
                    &source,
                    layer,
                    &hidden,
                    AttentionCache {
                        plain: &mut kv[index],
                        packed: &mut packed_kv[index],
                        mixed_workspace: mixed_workspaces
                            .as_mut()
                            .map(|workspaces| &mut workspaces[stage]),
                    },
                    &mut phase,
                )?
            };
            writeln!(report, "kv_reuse\t{turn}\t{index}\t{cached_before}\t1")?;
            let post_norm_start = Instant::now();
            let mut after_attention = engine.uninit(HIDDEN)?;
            engine.add(&hidden, &attention, &mut after_attention, HIDDEN)?;
            drop((hidden, attention));
            let post_norm_name = format!("model.layers.{index}.post_attention_layernorm.weight");
            let post_norm = if let Some(weights) = text_layer {
                normalized_with_weight(
                    engine,
                    &after_attention,
                    &weights.post_norm,
                    &post_norm_name,
                    &layer.pre_mlp_norm,
                )?
            } else {
                normalized(
                    engine,
                    &model,
                    &after_attention,
                    &post_norm_name,
                    &layer.pre_mlp_norm,
                )?
            };
            record_phase(
                &mut report,
                engine,
                profile_phases,
                turn,
                index,
                "post_attention_norm",
                post_norm_start,
            )?;
            let mlp_start = Instant::now();
            let mlp = match &layer.mlp {
                MlpPlan::Dense(dense) if index == 0 => {
                    if let Some(weights) = text_layer {
                        dense_mlp_token_resident(
                            engine,
                            &post_norm,
                            dense,
                            weights
                                .dense
                                .as_ref()
                                .ok_or("MiMo resident dense layer missing")?,
                        )?
                    } else {
                        dense_mlp_token(engine, &source, &post_norm, dense)?
                    }
                }
                MlpPlan::Moe(moe) if index > 0 => {
                    let result = if grouped_moe {
                        grouped_layers[index]
                            .as_ref()
                            .ok_or("MiMo grouped MoE layer was not loaded")?
                            .token(engine, &post_norm, moe, &pinned)?
                    } else if resident_moe {
                        resident_layers[index]
                            .as_ref()
                            .ok_or("MiMo resident MoE layer was not loaded")?
                            .token(engine, &post_norm, moe, &pinned)?
                    } else {
                        source_moe_token(engine, &pinned, index, &post_norm, moe)?
                    };
                    writeln!(
                        report,
                        "selected_experts\t{turn}\t{index}\t{:?}",
                        result.selected
                    )?;
                    writeln!(
                        report,
                        "routing_weights\t{turn}\t{index}\t{:?}",
                        result.weights
                    )?;
                    result.output
                }
                _ => return Err(format!("layer {index}: unsupported MiMo MLP plan").into()),
            };
            record_phase(
                &mut report,
                engine,
                profile_phases,
                turn,
                index,
                "mlp",
                mlp_start,
            )?;
            let residual_start = Instant::now();
            let mut after_mlp = engine.uninit(HIDDEN)?;
            engine.add(&after_attention, &mlp, &mut after_mlp, HIDDEN)?;
            hidden = after_mlp;
            let observed = engine.dtoh(&hidden)?;
            if observed.len() != HIDDEN || observed.iter().any(|value| !value.is_finite()) {
                return Err(format!("turn {turn} layer {index}: non-finite hidden row").into());
            }
            record_phase(
                &mut report,
                engine,
                profile_phases,
                turn,
                index,
                "residual_probe",
                residual_start,
            )?;
            writeln!(
                report,
                "layer_ms\t{turn}\t{index}\t{stage}\t{:.3}",
                before.elapsed().as_secs_f64() * 1000.0
            )?;
            eprintln!("MiMo GPU turn {turn} layer {index} stage {stage} complete");
        }
        let last = &engines[1];
        let head_weights = if resident_text {
            Some(resident_head.as_ref().ok_or("MiMo head was not resident")?)
        } else {
            None
        };
        let head_norm_start = Instant::now();
        let final_norm = if let Some(weights) = head_weights {
            normalized_with_weight(
                last,
                &hidden,
                &weights.norm,
                "model.norm.weight",
                &plan.output_norm,
            )?
        } else {
            normalized(
                last,
                &model,
                &hidden,
                "model.norm.weight",
                &plan.output_norm,
            )?
        };
        record_phase(
            &mut report,
            last,
            profile_phases,
            turn,
            LAYERS,
            "head_norm",
            head_norm_start,
        )?;
        let head_start = Instant::now();
        let streamed_head = if head_weights.is_none() {
            Some(bf16_matrix(last, &model, "lm_head.weight", VOCAB, HIDDEN)?)
        } else {
            None
        };
        let head = if let Some(weights) = head_weights {
            &weights.weight
        } else {
            streamed_head
                .as_ref()
                .ok_or("MiMo streamed head is missing")?
        };
        let logits_gpu = last.matmul(head, &final_norm, 1)?;
        let logits = last.dtoh(&logits_gpu)?;
        record_phase(
            &mut report,
            last,
            profile_phases,
            turn,
            LAYERS,
            "head_project",
            head_start,
        )?;
        if logits.len() != VOCAB || logits.iter().any(|value| !value.is_finite()) {
            return Err("MiMo GPU output logits are non-finite or wrong width".into());
        }
        let mut argmax = 0usize;
        for index in 1..VOCAB {
            if logits[index].total_cmp(&logits[argmax]).is_gt() {
                argmax = index;
            }
        }
        writeln!(report, "argmax_turn\t{turn}\t{argmax}")?;
        writeln!(
            report,
            "turn_wall_ms\t{turn}\t{:.3}",
            turn_start.elapsed().as_secs_f64() * 1000.0
        )?;
        current_token = argmax;
        last_argmax = Some(argmax);
        last_logits = Some(logits);
    }
    let logits = last_logits.ok_or("MiMo token runner produced no logit row")?;
    let argmax = last_argmax.ok_or("MiMo token runner produced no argmax")?;
    writeln!(report, "vocab\t{VOCAB}")?;
    writeln!(report, "argmax\t{argmax}")?;
    writeln!(report, "wall_s\t{:.3}", started.elapsed().as_secs_f64())?;
    for (index, value) in logits.iter().enumerate() {
        writeln!(report, "logit\t{index}\t{:08x}", value.to_bits())?;
    }
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&staged_path)?;
    file.write_all(report.as_bytes())?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(staged_path, report_path)?;
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("mimo_source_gpu_token: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_text_plan_passes_before_gpu_and_changed_state_refuses() {
        let fixture = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../memra-gguf/src/model_packs/mimo_v2/fixtures/config.json"
        ));
        let config = ModelConfig::from_hf(&HfConfig::parse(fixture));
        let pack = model_packs::by_alias("mimo_v2_source").unwrap();
        let mut plan = pack.compile_plan(&config).unwrap();
        validate_plan(&plan).unwrap();
        plan.layers[1].state = StatePlan::KvCache {
            key_width: 1536,
            value_width: 1024,
        };
        assert!(validate_plan(&plan).is_err());
        plan = pack.compile_plan(&config).unwrap();
        plan.partition_boundaries.retain(|&cut| cut != STAGE_CUT);
        assert!(validate_plan(&plan).is_err());
    }
}
