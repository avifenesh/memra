//! Deterministic MiMo attention component comparison on one GPU at a time.
//! Run separately on both cards. This does not qualify a model request.

use memra_engine::Engine;
use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::model_plan::{AttentionPlan, ModelPlan};
use sha2::{Digest, Sha256};

type Fail = Box<dyn std::error::Error>;

const HEADS: usize = 64;
const QK: usize = 192;
const V: usize = 128;

fn cpu_decode(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    sink: Option<&[f32]>,
    seq: usize,
    kv_heads: usize,
    window: usize,
) -> Vec<f32> {
    let start = if window == 0 {
        0
    } else {
        seq.saturating_sub(window)
    };
    let mut result = vec![0.0; HEADS * V];
    for head in 0..HEADS {
        let kv_head = head / (HEADS / kv_heads);
        let scores: Vec<f64> = (start..seq)
            .map(|token| {
                let k_base = (token * kv_heads + kv_head) * QK;
                let dot: f64 = (0..QK)
                    .map(|dim| (query[head * QK + dim] as f64) * (key[k_base + dim] as f64))
                    .sum();
                dot / (QK as f64).sqrt()
            })
            .collect();
        let maximum = scores
            .iter()
            .copied()
            .chain(sink.map(|bias| bias[head] as f64))
            .fold(f64::NEG_INFINITY, f64::max);
        let denominator: f64 = scores
            .iter()
            .map(|score| (score - maximum).exp())
            .sum::<f64>()
            + sink.map_or(0.0, |bias| ((bias[head] as f64) - maximum).exp());
        for (offset, score) in scores.iter().enumerate() {
            let weight = (score - maximum).exp() / denominator;
            let v_base = ((start + offset) * kv_heads + kv_head) * V;
            for dim in 0..V {
                result[head * V + dim] += (weight * value[v_base + dim] as f64) as f32;
            }
        }
    }
    result
}

fn check(
    engine: &Engine,
    plan: &AttentionPlan,
    seq: usize,
    window: usize,
    label: &str,
) -> Result<(), Fail> {
    let kv_heads = match plan {
        AttentionPlan::Full(full) => full.kv_heads as usize,
        AttentionPlan::SlidingWindow { attention, .. } => attention.kv_heads as usize,
        _ => return Err("MiMo gate requires full or sliding attention".into()),
    };
    let query: Vec<f32> = (0..HEADS * QK)
        .map(|index| ((index * 13 % 29) as f32 - 14.0) / 32.0)
        .collect();
    let key: Vec<f32> = (0..seq * kv_heads * QK)
        .map(|index| ((index * 7 % 31) as f32 - 15.0) / 24.0)
        .collect();
    let mut value: Vec<f32> = (0..seq * kv_heads * V)
        .map(|index| ((index * 11 % 37) as f32 - 18.0) / 9.0)
        .collect();
    if seq > 128 {
        for element in &mut value[..kv_heads * V] {
            *element += 19.0;
        }
    }
    let sink = (window == 128).then(|| {
        (0..HEADS)
            .map(|head| (head as f32 % 7.0) / 5.0 - 0.4)
            .collect::<Vec<_>>()
    });
    let expected = cpu_decode(&query, &key, &value, sink.as_deref(), seq, kv_heads, window);
    let q_gpu = engine.htod(&query)?;
    let k_gpu = engine.htod(&key)?;
    let v_gpu = engine.htod(&value)?;
    let sink_gpu = sink.as_ref().map(|v| engine.htod(v)).transpose()?;
    let actual = engine.dtoh(&engine.mimo_sink_decode(
        &q_gpu,
        &k_gpu,
        &v_gpu,
        sink_gpu.as_ref(),
        seq,
        plan,
    )?)?;
    if actual.len() != expected.len() || actual.iter().any(|value| !value.is_finite()) {
        return Err(format!("{label} seq={seq}: non-finite or wrong-size GPU output").into());
    }
    let max_abs = expected
        .iter()
        .zip(actual.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("{label}\t{seq}\t{max_abs:.8}");
    if max_abs > 0.0002 {
        return Err(format!("{label} seq={seq}: max_abs {max_abs} > 0.0002").into());
    }
    Ok(())
}

fn run() -> Result<(), Fail> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: mimo_sink_attn_gate <source_config.json> <gpu_index>".into());
    }
    let config_bytes = std::fs::read(&args[0])?;
    let config_digest = format!("{:x}", Sha256::digest(&config_bytes));
    if config_digest != "61bea4a0f7a0dd8969f8cae528761e26b697dd12ff63e98804c3f0945492e621" {
        return Err("MiMo gate requires the pinned source config".into());
    }
    let gpu_index: usize = args[1].parse()?;
    let config_text = String::from_utf8(config_bytes)?;
    let config = ModelConfig::from_hf(&HfConfig::try_parse(&config_text)?);
    let plan = ModelPlan::compile(&config)?;
    let full = &plan.layers[0].attention;
    let AttentionPlan::Full(_) = full else {
        return Err("pinned layer 0 is not full attention".into());
    };
    let sliding = &plan.layers[1].attention;
    let AttentionPlan::SlidingWindow { window, .. } = sliding else {
        return Err("pinned layer 1 is not sliding attention".into());
    };
    if *window != 128 {
        return Err("pinned sliding window differs from 128".into());
    }
    let engine = Engine::new(gpu_index)?;
    println!("gpu_index\t{gpu_index}");
    println!("layer\tseq\tmax_abs");
    for seq in [1, 3, 129, 256] {
        check(&engine, full, seq, 0, "full")?;
        check(&engine, sliding, seq, 128, "sliding")?;
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("mimo_sink_attn_gate: {error}");
        std::process::exit(1);
    }
}
