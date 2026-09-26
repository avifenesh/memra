//! Component gate for MiMo's checkpoint-sharded block-FP8 QKV projection.
//! This is a diagnostic on one GPU at a time, not a MiMo serving path.

use std::fmt::Write as _;
use std::io::Write as _;
use std::path::Path;

use memra_engine::Engine;
use memra_engine::model::GpuTensor;
use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::model_packs::mimo_v2::inspect_pinned_source_headers;
use memra_gguf::safetensors::StModel;
use memra_gguf::source::{SafetensorsSource, TensorSource};

type Fail = Box<dyn std::error::Error>;

fn main() {
    if let Err(error) = run() {
        eprintln!("mimo_qkv_shard_gate: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Fail> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("usage: mimo_qkv_shard_gate <source_dir> <gpu_index> <report.tsv>".into());
    }
    let dir = Path::new(&args[0]);
    let gpu_index: usize = args[1].parse()?;
    let report = Path::new(&args[2]);
    if report.exists() {
        return Err(format!("refusing to overwrite {}", report.display()).into());
    }

    let config_text = std::fs::read_to_string(dir.join("config.json"))?;
    let config = ModelConfig::from_hf(&HfConfig::try_parse(&config_text)?);
    let model = StModel::open(dir)?;
    let headers = model
        .names()
        .map(|name| (name.clone(), model.info(name).unwrap().clone()))
        .collect();
    let bound = inspect_pinned_source_headers(&config, &headers)?;
    eprintln!(
        "pinned source headers bound {} semantic tensors",
        bound.tensors.len()
    );
    drop(bound);
    drop(headers);
    drop(model);

    let source = SafetensorsSource::open(dir)?;
    let engine = Engine::new(gpu_index)?;
    let input: Vec<f32> = (0..config.n_embd as usize)
        .map(|index| match index % 3 {
            0 => -1.0,
            1 => 0.0,
            _ => 1.0,
        })
        .collect();
    let input_device = engine.htod(&input)?;
    let mut text = String::from("format\tmemra-mimo-qkv-shard-v1\n");
    writeln!(text, "gpu_index\t{gpu_index}")?;
    writeln!(text, "input_width\t{}", input.len())?;

    for layer in [0u32, 1u32] {
        let name = format!("model.layers.{layer}.self_attn.qkv_proj.weight");
        let views = source
            .find_fp8_mimo_qkv_shards(&name)
            .ok_or_else(|| format!("no four-shard native view for {name}"))?;
        if views.len() != 4 {
            return Err(format!("{name}: expected four source shards, got {}", views.len()).into());
        }
        let (dequantized, _) = source
            .dequant_f32_hf(&name)
            .ok_or_else(|| format!("{name}: no CPU dequant reference"))?;
        let rows_per_shard = views[0].out_f;
        for (shard, view) in views.iter().enumerate() {
            if view.out_f != rows_per_shard || view.in_f != input.len() {
                return Err(format!("{name} shard {shard}: inconsistent geometry").into());
            }
            let weight = GpuTensor::load_mimo_fp8_qkv_shard(&engine, view)?;
            let actual = engine.dtoh(&engine.matmul(&weight, &input_device, 1)?)?;
            if actual.len() != rows_per_shard {
                return Err(format!("{name} shard {shard}: output width mismatch").into());
            }
            let mut max_abs = 0.0f64;
            let mut sum_abs = 0.0f64;
            let mut dot = 0.0f64;
            let mut cpu_norm = 0.0f64;
            let mut gpu_norm = 0.0f64;
            for (row, &observed) in actual.iter().enumerate() {
                if !observed.is_finite() {
                    return Err(
                        format!("{name} shard {shard} row {row}: non-finite GPU output").into(),
                    );
                }
                let offset = (shard * rows_per_shard + row) * input.len();
                let mut expected = 0.0f32;
                for (value, weight) in input.iter().zip(&dequantized[offset..offset + input.len()])
                {
                    expected = value.mul_add(*weight, expected);
                }
                if !expected.is_finite() {
                    return Err(
                        format!("{name} shard {shard} row {row}: non-finite CPU output").into(),
                    );
                }
                let error = (expected as f64 - observed as f64).abs();
                max_abs = max_abs.max(error);
                sum_abs += error;
                dot += expected as f64 * observed as f64;
                cpu_norm += (expected as f64).powi(2);
                gpu_norm += (observed as f64).powi(2);
            }
            if cpu_norm <= 0.0 || gpu_norm <= 0.0 {
                return Err(format!("{name} shard {shard}: zero output norm").into());
            }
            let cosine = dot / (cpu_norm.sqrt() * gpu_norm.sqrt());
            writeln!(
                text,
                "shard\t{layer}\t{shard}\t{}\t{max_abs:.9}\t{:.9}\t{cosine:.9}",
                rows_per_shard,
                sum_abs / rows_per_shard as f64
            )?;
            eprintln!(
                "layer {layer} shard {shard}: rows={rows_per_shard} max_abs={max_abs:.6} cosine={cosine:.8}"
            );
        }
    }
    let temporary = report.with_extension("tsv.writing");
    let mut staged = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    staged.write_all(text.as_bytes())?;
    staged.sync_all()?;
    drop(staged);
    std::fs::rename(&temporary, report)?;
    Ok(())
}
