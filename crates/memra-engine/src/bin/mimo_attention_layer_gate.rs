//! Offline, source-backed MiMo one-layer attention component gate.
//! No residual, MLP, request, or serving capability is checked here.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::Path;

use memra_engine::Engine;
use memra_engine::model::GpuTensor;
use memra_gguf::GgmlType;
use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::model_packs::mimo_v2::inspect_pinned_source_headers;
use memra_gguf::model_plan::{AttentionPlan, ModelPlan, NormKind, RopeFactors, WeightTransform};
use memra_gguf::safetensors::StModel;
use memra_gguf::source::{SafetensorsSource, TensorSource};
use sha2::{Digest, Sha256};

type Fail = Box<dyn std::error::Error>;

const HIDDEN: usize = 4096;
const HEADS: usize = 64;
const QK: usize = 192;
const VALUE: usize = 128;
const POSITION: i32 = 1;

fn main() {
    if let Err(error) = run() {
        eprintln!("mimo_attention_layer_gate: {error}");
        std::process::exit(1);
    }
}

fn source_vector(source: &SafetensorsSource, name: &str, len: usize) -> Result<Vec<f32>, Fail> {
    let (values, shape) = source
        .dequant_f32_hf(name)
        .ok_or_else(|| format!("{name}: source dequant unavailable"))?;
    if shape != [len as u64] || values.len() != len || values.iter().any(|x| !x.is_finite()) {
        return Err(format!("{name}: wrong shape or non-finite source value").into());
    }
    Ok(values)
}

fn source_matrix(
    source: &SafetensorsSource,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>, Fail> {
    let (values, shape) = source
        .dequant_f32_hf(name)
        .ok_or_else(|| format!("{name}: source dequant unavailable"))?;
    if shape != [cols as u64, rows as u64]
        || values.len() != rows * cols
        || values.iter().any(|x| !x.is_finite())
    {
        return Err(format!("{name}: wrong shape or non-finite source value").into());
    }
    Ok(values)
}

fn cpu_norm(input: &[f32], weight: &[f32], epsilon: f32) -> Vec<f32> {
    let mean_square =
        input.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / input.len() as f64;
    let inv = (mean_square + epsilon as f64).sqrt().recip();
    input
        .iter()
        .zip(weight)
        .map(|(&x, &w)| (x as f64 * inv * w as f64) as f32)
        .collect()
}

fn cpu_matvec(weight: &[f32], rows: usize, input: &[f32]) -> Vec<f32> {
    weight
        .chunks_exact(input.len())
        .take(rows)
        .map(|row| {
            row.iter()
                .zip(input)
                .map(|(&w, &x)| (w as f64) * (x as f64))
                .sum::<f64>() as f32
        })
        .collect()
}

fn cpu_rope(vector: &mut [f32], heads: usize, dims: usize, base: f32, position: i32) {
    let half = dims / 2;
    for head in 0..heads {
        let row = &mut vector[head * QK..(head + 1) * QK];
        for pair in 0..half {
            let theta = position as f64 / (base as f64).powf(2.0 * pair as f64 / dims as f64);
            let (sin, cos) = theta.sin_cos();
            let left = row[pair] as f64;
            let right = row[pair + half] as f64;
            row[pair] = (left * cos - right * sin) as f32;
            row[pair + half] = (left * sin + right * cos) as f32;
        }
    }
}

fn cpu_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    sink: Option<&[f32]>,
    kv_heads: usize,
) -> (Vec<f32>, Option<(f64, f64)>) {
    let mut result = vec![0.0; HEADS * VALUE];
    let mut sink_range = sink.map(|_| (f64::INFINITY, 0.0f64));
    for head in 0..HEADS {
        let kv_head = head / (HEADS / kv_heads);
        let score = query[head * QK..(head + 1) * QK]
            .iter()
            .zip(&key[kv_head * QK..(kv_head + 1) * QK])
            .map(|(&q, &k)| q as f64 * k as f64)
            .sum::<f64>()
            / (QK as f64).sqrt();
        let sink_probability = sink.map(|bias| {
            let delta = bias[head] as f64 - score;
            if delta >= 0.0 {
                1.0 / (1.0 + (-delta).exp())
            } else {
                delta.exp() / (1.0 + delta.exp())
            }
        });
        if let (Some(probability), Some(range)) = (sink_probability, sink_range.as_mut()) {
            range.0 = range.0.min(probability);
            range.1 = range.1.max(probability);
        }
        let weight = 1.0 - sink_probability.unwrap_or(0.0);
        for dim in 0..VALUE {
            result[head * VALUE + dim] = (weight * value[kv_head * VALUE + dim] as f64) as f32;
        }
    }
    (result, sink_range)
}

fn compare(
    report: &mut String,
    layer: u32,
    stage: &str,
    expected: &[f32],
    actual: &[f32],
    atol: f64,
    rtol: f64,
) -> Result<(), Fail> {
    if actual.len() != expected.len() || actual.iter().any(|x| !x.is_finite()) {
        return Err(format!("layer {layer} {stage}: wrong shape or non-finite GPU value").into());
    }
    if expected.iter().any(|x| !x.is_finite()) {
        return Err(format!("layer {layer} {stage}: non-finite CPU value").into());
    }
    let mut max_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut failures = 0usize;
    for (&reference, &observed) in expected.iter().zip(actual) {
        let error = (reference as f64 - observed as f64).abs();
        max_abs = max_abs.max(error);
        sum_sq += error * error;
        if error > atol + rtol * (reference as f64).abs() {
            failures += 1;
        }
    }
    let rms = (sum_sq / expected.len() as f64).sqrt();
    writeln!(
        report,
        "vector\t{layer}\t{stage}\t{}\t{max_abs:.9}\t{rms:.9}\t{failures}\t{atol:.6}\t{rtol:.6}",
        expected.len()
    )?;
    if failures != 0 {
        return Err(format!("layer {layer} {stage}: {failures} values exceed tolerance").into());
    }
    Ok(())
}

fn run() -> Result<(), Fail> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err(
            "usage: mimo_attention_layer_gate <source_dir> <gpu_index> <report.tsv>".into(),
        );
    }
    let dir = Path::new(&args[0]);
    let gpu_index: usize = args[1].parse()?;
    let report_path = Path::new(&args[2]);
    if report_path.exists() {
        return Err(format!("refusing to overwrite {}", report_path.display()).into());
    }

    let config_bytes = std::fs::read(dir.join("config.json"))?;
    let config_digest = format!("{:x}", Sha256::digest(&config_bytes));
    if config_digest != "61bea4a0f7a0dd8969f8cae528761e26b697dd12ff63e98804c3f0945492e621" {
        return Err(format!("pinned source config digest changed: {config_digest}").into());
    }
    let config_text = String::from_utf8(config_bytes)?;
    let config = ModelConfig::from_hf(&HfConfig::try_parse(&config_text)?);
    let plan = ModelPlan::compile(&config)?;
    let model = StModel::open(dir)?;
    let headers = model
        .names()
        .map(|name| (name.clone(), model.info(name).unwrap().clone()))
        .collect::<BTreeMap<_, _>>();
    let bound = inspect_pinned_source_headers(&config, &headers)?;
    let bound_count = bound.tensors.len();
    drop((bound, headers, model));
    if config.n_embd as usize != HIDDEN || plan.layers.len() < 2 {
        return Err("pinned MiMo hidden width or layer count changed".into());
    }

    let source = SafetensorsSource::open(dir)?;
    let engine = Engine::new(gpu_index)?;
    let input: Vec<f32> = (0..HIDDEN)
        .map(|index| match index % 3 {
            0 => -1.0,
            1 => 0.0,
            _ => 1.0,
        })
        .collect();
    let input_device = engine.htod(&input)?;
    let position_device = engine.htod_i32(&[POSITION])?;
    let mut report = String::from("format\tmemra-mimo-attention-layer-v1\n");
    writeln!(
        report,
        "target_source\tXiaomiMiMo/MiMo-V2.6-Flash-RL@3b38d063180c3e4aed9691fdc735f3d10b266ee4"
    )?;
    writeln!(report, "config_sha256\t{config_digest}")?;
    writeln!(
        report,
        "source_header_digest\t5ebbdd27e45716b805fc2bdf115c8345b4bfc03c6012b860f76b6b222c758aee"
    )?;
    writeln!(report, "bound_semantic_tensors\t{bound_count}")?;
    writeln!(report, "gpu_index\t{gpu_index}")?;
    writeln!(report, "position\t{POSITION}")?;
    writeln!(report, "sequence_length\t1")?;
    writeln!(report, "input_pattern\tminus_one_zero_one")?;
    writeln!(
        report,
        "vector_columns\tlayer\tstage\tlength\tmax_abs\trms\tfailures\tatol\trtol"
    )?;

    for layer in [0u32, 1u32] {
        let layer_plan = &plan.layers[layer as usize];
        let (attention, kv_heads, kind) = match &layer_plan.attention {
            AttentionPlan::Full(full) if layer == 0 && full.kv_heads == 4 => {
                (full, 4usize, "global")
            }
            AttentionPlan::SlidingWindow { attention, window }
                if layer == 1 && *window == 128 && attention.kv_heads == 8 =>
            {
                (attention, 8usize, "sliding")
            }
            _ => return Err(format!("layer {layer}: pinned attention type changed").into()),
        };
        let rope = attention.rope;
        if rope.dimensions != 64
            || !matches!(rope.factors, RopeFactors::None)
            || !rope.base.is_finite()
            || rope.base <= 1.0
            || layer_plan.pre_attention_norm.kind != NormKind::Rms
            || layer_plan.pre_attention_norm.weight_transform != WeightTransform::Identity
        {
            return Err(format!("layer {layer}: unsupported source norm or RoPE").into());
        }
        writeln!(report, "layer\t{layer}\t{kind}\t{}\t64", rope.base)?;
        let prefix = format!("model.layers.{layer}");
        let norm_name = format!("{prefix}.input_layernorm.weight");
        let norm_weight = source_vector(&source, &norm_name, HIDDEN)?;
        let norm_weight_device = engine.htod(&norm_weight)?;
        let mut norm_device = engine.uninit(HIDDEN)?;
        engine.rms_norm(
            &input_device,
            &norm_weight_device,
            &mut norm_device,
            HIDDEN,
            1,
            layer_plan.pre_attention_norm.epsilon,
        )?;
        let norm_cpu = cpu_norm(&input, &norm_weight, layer_plan.pre_attention_norm.epsilon);
        compare(
            &mut report,
            layer,
            "pre_attention_norm",
            &norm_cpu,
            &engine.dtoh(&norm_device)?,
            0.0001,
            0.0001,
        )?;

        let qkv_name = format!("{prefix}.self_attn.qkv_proj.weight");
        let shard_views = source
            .find_fp8_mimo_qkv_shards(&qkv_name)
            .ok_or_else(|| format!("{qkv_name}: native four-shard view unavailable"))?;
        if shard_views.len() != 4 {
            return Err(format!("{qkv_name}: expected exactly four native shards").into());
        }
        let shard_rows = 3072 + kv_heads / 4 * (QK + VALUE);
        let qkv_weights = source_matrix(&source, &qkv_name, 4 * shard_rows, HIDDEN)?;
        let mut projections = Vec::with_capacity(4);
        let mut cpu_projections = Vec::with_capacity(4);
        for (shard, view) in shard_views.iter().enumerate() {
            if view.out_f != shard_rows || view.in_f != HIDDEN {
                return Err(format!("{qkv_name} shard {shard}: wrong shape").into());
            }
            let weight = GpuTensor::load_mimo_fp8_qkv_shard(&engine, view)?;
            let projected = engine.matmul(&weight, &norm_device, 1)?;
            let cpu = cpu_matvec(
                &qkv_weights[shard * shard_rows * HIDDEN..(shard + 1) * shard_rows * HIDDEN],
                shard_rows,
                &norm_cpu,
            );
            compare(
                &mut report,
                layer,
                &format!("qkv_shard_{shard}"),
                &cpu,
                &engine.dtoh(&projected)?,
                0.02,
                0.01,
            )?;
            projections.push(projected);
            cpu_projections.push(cpu);
        }
        let mut gathered = engine.mimo_gather_qkv(
            [
                &projections[0],
                &projections[1],
                &projections[2],
                &projections[3],
            ],
            1,
            attention,
        )?;
        let mut q_cpu = Vec::with_capacity(HEADS * QK);
        let mut k_cpu = Vec::with_capacity(kv_heads * QK);
        let mut v_cpu = Vec::with_capacity(kv_heads * VALUE);
        for projected in &cpu_projections {
            q_cpu.extend_from_slice(&projected[..3072]);
            k_cpu.extend_from_slice(&projected[3072..3072 + kv_heads / 4 * QK]);
            v_cpu.extend(
                projected[3072 + kv_heads / 4 * QK..]
                    .iter()
                    .map(|v| v * 0.707f32),
            );
        }
        compare(
            &mut report,
            layer,
            "gather_q",
            &q_cpu,
            &engine.dtoh(&gathered.query)?,
            0.02,
            0.01,
        )?;
        compare(
            &mut report,
            layer,
            "gather_k",
            &k_cpu,
            &engine.dtoh(&gathered.key)?,
            0.02,
            0.01,
        )?;
        compare(
            &mut report,
            layer,
            "gather_v_scaled",
            &v_cpu,
            &engine.dtoh(&gathered.value)?,
            0.02,
            0.01,
        )?;

        engine.rope_neox(
            &mut gathered.query,
            &position_device,
            QK,
            rope.dimensions as usize,
            HEADS,
            1,
            rope.base,
            1.0,
        )?;
        engine.rope_neox(
            &mut gathered.key,
            &position_device,
            QK,
            rope.dimensions as usize,
            kv_heads,
            1,
            rope.base,
            1.0,
        )?;
        cpu_rope(&mut q_cpu, HEADS, 64, rope.base, POSITION);
        cpu_rope(&mut k_cpu, kv_heads, 64, rope.base, POSITION);
        compare(
            &mut report,
            layer,
            "rope_q",
            &q_cpu,
            &engine.dtoh(&gathered.query)?,
            0.02,
            0.01,
        )?;
        compare(
            &mut report,
            layer,
            "rope_k",
            &k_cpu,
            &engine.dtoh(&gathered.key)?,
            0.02,
            0.01,
        )?;

        let sink_cpu = if layer == 1 {
            Some(source_vector(
                &source,
                &format!("{prefix}.self_attn.attention_sink_bias"),
                HEADS,
            )?)
        } else {
            None
        };
        let sink_device = sink_cpu.as_ref().map(|v| engine.htod(v)).transpose()?;
        let (context_cpu, sink_range) =
            cpu_attention(&q_cpu, &k_cpu, &v_cpu, sink_cpu.as_deref(), kv_heads);
        if let Some((minimum, maximum)) = sink_range {
            if !minimum.is_finite() || !maximum.is_finite() || maximum <= 1e-6 {
                return Err(format!("layer {layer}: sink was numerically inactive").into());
            }
            writeln!(
                report,
                "sink_probability\t{layer}\t{minimum:.9}\t{maximum:.9}"
            )?;
        }
        let context_device = engine.mimo_sink_decode(
            &gathered.query,
            &gathered.key,
            &gathered.value,
            sink_device.as_ref(),
            1,
            &layer_plan.attention,
        )?;
        compare(
            &mut report,
            layer,
            "sink_decode_context",
            &context_cpu,
            &engine.dtoh(&context_device)?,
            0.03,
            0.02,
        )?;

        let output_name = format!("{prefix}.self_attn.o_proj.weight");
        let output_raw = source
            .raw_hf(&output_name)
            .ok_or_else(|| format!("{output_name}: source BF16 view unavailable"))?;
        if output_raw.ggml_type != GgmlType::BF16
            || output_raw.ne != [HEADS as u64 * VALUE as u64, HIDDEN as u64]
            || output_raw.bytes.len() != HIDDEN * HEADS * VALUE * 2
        {
            return Err(format!("{output_name}: wrong source BF16 shape or storage").into());
        }
        let output_weights = source_matrix(&source, &output_name, HIDDEN, HEADS * VALUE)?;
        let output_weight = GpuTensor::FloatBf16 {
            data: engine.htod_bytes(&output_raw.bytes)?,
            ne: output_raw.ne,
        };
        let output_device = engine.matmul(&output_weight, &context_device, 1)?;
        let output_cpu = cpu_matvec(&output_weights, HIDDEN, &context_cpu);
        compare(
            &mut report,
            layer,
            "output_projection",
            &output_cpu,
            &engine.dtoh(&output_device)?,
            0.05,
            0.02,
        )?;
    }
    let temporary = report_path.with_extension("tsv.writing");
    let mut staged = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    staged.write_all(report.as_bytes())?;
    staged.sync_all()?;
    drop(staged);
    std::fs::rename(temporary, report_path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_rope_rotates_at_nonzero_position_and_preserves_tail() {
        let mut row = vec![0.0; QK];
        row[0] = 1.0;
        row[32] = 2.0;
        row[64] = 3.0;
        cpu_rope(&mut row, 1, 64, 10_000.0, 1);
        assert!((row[0] - (1.0f32.cos() - 2.0 * 1.0f32.sin())).abs() < 1e-6);
        assert!((row[32] - (1.0f32.sin() + 2.0 * 1.0f32.cos())).abs() < 1e-6);
        assert_eq!(row[64], 3.0);
    }

    #[test]
    fn one_token_sink_contributes_to_denominator() {
        let q = vec![0.0; HEADS * QK];
        let k = vec![0.0; 8 * QK];
        let v = vec![2.0; 8 * VALUE];
        let sink = vec![0.0; HEADS];
        let (actual, range) = cpu_attention(&q, &k, &v, Some(&sink), 8);
        assert_eq!(actual, vec![1.0; HEADS * VALUE]);
        assert_eq!(range, Some((0.5, 0.5)));
    }

    #[test]
    fn scalar_projection_uses_source_row_order() {
        let weight = [1.0, 2.0, 3.0, -1.0, 0.5, 4.0];
        assert_eq!(cpu_matvec(&weight, 2, &[2.0, -1.0, 3.0]), [9.0, 9.5]);
    }
}
