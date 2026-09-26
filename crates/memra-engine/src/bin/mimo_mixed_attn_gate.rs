//! Deterministic component gate for packed MiMo global attention.
//! A passing component check does not qualify MiMo serving.

use std::time::Instant;

use memra_engine::Engine;
use memra_engine::mimo_mixed_attn_ffi::MiMoMixedAttentionWorkspace;
use memra_gguf::GgmlType;
use memra_gguf::dequant::dequantize;
use memra_gguf::nvfp4_repack::{f32_to_nvfp4, f32_to_q8_0};
use sha2::{Digest, Sha256};

type Fail = Box<dyn std::error::Error>;
const HEADS: usize = 64;
const KV_HEADS: usize = 4;
const QK: usize = 192;
const VALUE: usize = 128;

fn cpu_oracle(q: &[f32], k: &[f32], v: &[f32], seq: usize) -> Vec<f32> {
    let mut output = vec![0.0; HEADS * VALUE];
    for head in 0..HEADS {
        let kv_head = head / (HEADS / KV_HEADS);
        let scores: Vec<f64> = (0..seq)
            .map(|token| {
                let base = (token * KV_HEADS + kv_head) * QK;
                let dot: f64 = (0..QK)
                    .map(|i| f64::from(q[head * QK + i]) * f64::from(k[base + i]))
                    .sum();
                dot / (QK as f64).sqrt()
            })
            .collect();
        let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = scores
            .iter()
            .map(|score| (score - max_score).exp())
            .collect();
        let denom: f64 = weights.iter().sum();
        for (token, weight) in weights.into_iter().enumerate() {
            let base = (token * KV_HEADS + kv_head) * VALUE;
            for dim in 0..VALUE {
                output[head * VALUE + dim] += (weight / denom * f64::from(v[base + dim])) as f32;
            }
        }
    }
    output
}

fn check_pattern(
    engine: &Engine,
    seq: usize,
    attention_only: bool,
    program: u8,
) -> Result<(), Fail> {
    let query: Vec<f32> = (0..HEADS * QK)
        .map(|i| ((i * 7 % 89) as f32 - 44.0) / 64.0)
        .collect();
    let key: Vec<f32> = (0..seq * KV_HEADS * QK)
        .map(|i| ((i * 11 % 97) as f32 - 48.0) / 96.0)
        .collect();
    let value: Vec<f32> = (0..seq * KV_HEADS * VALUE)
        .map(|i| ((i * 17 % 101) as f32 - 50.0) / 50.0)
        .collect();
    let key_bytes = f32_to_q8_0(&key);
    let value_bytes = f32_to_nvfp4(&value);
    let expected = cpu_oracle(
        &query,
        &dequantize(GgmlType::Q8_0, &key_bytes, key.len()),
        &dequantize(GgmlType::NVFP4, &value_bytes, value.len()),
        seq,
    );
    let query_gpu = engine.htod(&query)?;
    let key_gpu = engine.htod_bytes(&key_bytes)?;
    let value_gpu = if attention_only {
        engine.htod_bytes(&value_bytes)?
    } else {
        let value_f32_gpu = engine.htod(&value)?;
        let encoded_gpu = engine.mimo_nvfp4_encode_rows(&value_f32_gpu, VALUE)?;
        let encoded = engine.dtoh_u8(&encoded_gpu)?;
        if encoded != value_bytes {
            let mismatches = encoded
                .iter()
                .zip(&value_bytes)
                .filter(|(actual, expected)| actual != expected)
                .count();
            return Err(format!("seq={seq}: native NVFP4 byte mismatches {mismatches}").into());
        }
        let restored =
            engine.dtoh(&engine.mimo_nvfp4_decode_rows(&encoded_gpu, seq * KV_HEADS, VALUE)?)?;
        let decoded = dequantize(GgmlType::NVFP4, &value_bytes, value.len());
        if restored
            .iter()
            .zip(&decoded)
            .any(|(actual, expected)| actual.to_bits() != expected.to_bits())
        {
            return Err(format!("seq={seq}: native NVFP4 decode differs").into());
        }
        encoded_gpu
    };
    let mut workspace = match program {
        0 => MiMoMixedAttentionWorkspace::new(engine, seq)?,
        1 => MiMoMixedAttentionWorkspace::new_grouped(engine, seq)?,
        2 => MiMoMixedAttentionWorkspace::new_deep(engine, seq)?,
        _ => return Err("MiMo gate attention program is unavailable".into()),
    };
    let start = Instant::now();
    let actual = engine.dtoh(&engine.mimo_global_q8_nvfp4_decode(
        &query_gpu,
        &key_gpu,
        &value_gpu,
        seq,
        &mut workspace,
    )?)?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    if actual.len() != expected.len() || actual.iter().any(|value| !value.is_finite()) {
        return Err(format!("seq={seq}: invalid GPU output").into());
    }
    let max_abs = expected
        .iter()
        .zip(&actual)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("pattern\t{seq}\t{max_abs:.9e}\t{elapsed_ms:.4}");
    if max_abs > 0.0003 {
        return Err(format!("seq={seq}: max error {max_abs} exceeds 0.0003").into());
    }
    Ok(())
}

fn check_constant(engine: &Engine, seq: usize, program: u8) -> Result<(), Fail> {
    let query = vec![0.0f32; HEADS * QK];
    let key_row = f32_to_q8_0(&vec![0.0f32; KV_HEADS * QK]);
    let original_value: Vec<f32> = (0..KV_HEADS * VALUE)
        .map(|i| ((i * 13 % 67) as f32 - 33.0) / 31.0)
        .collect();
    let value_row = f32_to_nvfp4(&original_value);
    let decoded_row = dequantize(GgmlType::NVFP4, &value_row, original_value.len());
    let mut expected = vec![0.0f32; HEADS * VALUE];
    for head in 0..HEADS {
        let kv_head = head / (HEADS / KV_HEADS);
        expected[head * VALUE..(head + 1) * VALUE]
            .copy_from_slice(&decoded_row[kv_head * VALUE..(kv_head + 1) * VALUE]);
    }
    let key_bytes = key_row.repeat(seq);
    let value_bytes = value_row.repeat(seq);
    let query_gpu = engine.htod(&query)?;
    let key_gpu = engine.htod_bytes(&key_bytes)?;
    let value_gpu = engine.htod_bytes(&value_bytes)?;
    let mut workspace = match program {
        0 => MiMoMixedAttentionWorkspace::new(engine, seq)?,
        1 => MiMoMixedAttentionWorkspace::new_grouped(engine, seq)?,
        2 => MiMoMixedAttentionWorkspace::new_deep(engine, seq)?,
        _ => return Err("MiMo gate attention program is unavailable".into()),
    };
    let start = Instant::now();
    let actual = engine.dtoh(&engine.mimo_global_q8_nvfp4_decode(
        &query_gpu,
        &key_gpu,
        &value_gpu,
        seq,
        &mut workspace,
    )?)?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let max_abs = expected
        .iter()
        .zip(&actual)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("constant\t{seq}\t{max_abs:.9e}\t{elapsed_ms:.4}");
    if actual.iter().any(|value| !value.is_finite()) || max_abs > 0.0003 {
        return Err(format!("seq={seq}: constant-cache result diverged").into());
    }
    Ok(())
}

fn check_codec(engine: &Engine, width: usize) -> Result<(), Fail> {
    let values: Vec<f32> = (0..4 * width)
        .map(|i| {
            let base = [0.0, -0.0, 0.25, -0.5, 1.0, -3.0, 6.0, 30.0, 448.0];
            base[i % base.len()] * (1.0 + (i / width) as f32 * 0.125)
        })
        .collect();
    let expected = f32_to_nvfp4(&values);
    let input = engine.htod(&values)?;
    let encoded = engine.mimo_nvfp4_encode_rows(&input, width)?;
    let actual = engine.dtoh_u8(&encoded)?;
    if actual != expected {
        let mismatches = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
        let first: Vec<String> = actual
            .iter()
            .zip(&expected)
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .take(16)
            .map(|(i, (a, b))| format!("{i}:{a:02x}!={b:02x}"))
            .collect();
        return Err(format!(
            "width={width}: native NVFP4 byte mismatches {mismatches}; {}",
            first.join(",")
        )
        .into());
    }
    let expected_decoded = dequantize(GgmlType::NVFP4, &expected, values.len());
    let decoded = engine.dtoh(&engine.mimo_nvfp4_decode_rows(&encoded, 4, width)?)?;
    if decoded
        .iter()
        .zip(&expected_decoded)
        .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err(format!("width={width}: native NVFP4 decode differs").into());
    }
    println!("codec\t{width}\t0\t0");
    Ok(())
}

fn check_varied_million(engine: &Engine, program: u8) -> Result<(), Fail> {
    const SEQ: usize = 1_048_576;
    let query: Vec<f32> = (0..HEADS * QK)
        .map(|i| ((i * 29 % 113) as f32 - 56.0) / 56.0)
        .collect();
    let mut seed = 0x8d3f_1249u32;
    let mut next_byte = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (seed >> 24) as u8
    };
    let mut key_bytes = vec![0u8; SEQ * KV_HEADS * (QK / 32) * 34];
    for block in key_bytes.chunks_exact_mut(34) {
        block[..2].copy_from_slice(&0x2400u16.to_le_bytes()); // f16 1/64
        for code in &mut block[2..] {
            *code = ((next_byte() as i8) >> 1) as u8;
        }
    }
    let mut value_bytes = vec![0u8; SEQ * KV_HEADS * (VALUE / 64) * 36];
    for block in value_bytes.chunks_exact_mut(36) {
        block[..4].fill(0x30); // positive UE4M3 scale
        for codes in &mut block[4..] {
            *codes = next_byte();
        }
    }
    let query_gpu = engine.htod(&query)?;
    let key_gpu = engine.htod_bytes(&key_bytes)?;
    let value_gpu = engine.htod_bytes(&value_bytes)?;
    let mut baseline = MiMoMixedAttentionWorkspace::new(engine, SEQ)?;
    let mut candidate = match program {
        0 => MiMoMixedAttentionWorkspace::new(engine, SEQ)?,
        1 => MiMoMixedAttentionWorkspace::new_grouped(engine, SEQ)?,
        2 => MiMoMixedAttentionWorkspace::new_deep(engine, SEQ)?,
        _ => return Err("MiMo varied million program is unavailable".into()),
    };
    let timed = |workspace: &mut MiMoMixedAttentionWorkspace| -> Result<(Vec<f32>, f64), Fail> {
        let start = Instant::now();
        let output =
            engine.mimo_global_q8_nvfp4_decode(&query_gpu, &key_gpu, &value_gpu, SEQ, workspace)?;
        let values = engine.dtoh(&output)?;
        Ok((values, start.elapsed().as_secs_f64() * 1000.0))
    };
    let (reference, _) = timed(&mut baseline)?;
    let (actual, _) = timed(&mut candidate)?;
    if reference
        .iter()
        .chain(&actual)
        .any(|value| !value.is_finite())
    {
        return Err("MiMo varied million produced non-finite output".into());
    }
    let max_abs = reference
        .iter()
        .zip(&actual)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    if max_abs > 0.001 {
        return Err(format!("MiMo varied million candidate max error {max_abs}").into());
    }
    let mut times = [Vec::new(), Vec::new()];
    for iteration in 0..6 {
        for arm in [iteration % 2, 1 - iteration % 2] {
            let (_, elapsed) = if arm == 0 {
                timed(&mut baseline)?
            } else {
                timed(&mut candidate)?
            };
            times[arm].push(elapsed);
        }
    }
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    println!(
        "varied_million\t{SEQ}\t{max_abs:.9e}\tbaseline_ms={:.4}\tcandidate_ms={:.4}",
        mean(&times[0]),
        mean(&times[1]),
    );
    Ok(())
}

fn run() -> Result<(), Fail> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !(2..=6).contains(&args.len()) {
        return Err(
            "usage: mimo_mixed_attn_gate <pinned_config.json> <gpu_index> [--million] [--million-varied] [--attention-only] [--grouped | --deep]"
                .into(),
        );
    }
    let mut million = false;
    let mut million_varied = false;
    let mut attention_only = false;
    let mut grouped = false;
    let mut deep = false;
    for option in args.iter().skip(2) {
        match option.as_str() {
            "--million" if !million => million = true,
            "--million-varied" if !million_varied => million_varied = true,
            "--attention-only" if !attention_only => attention_only = true,
            "--grouped" if !grouped => grouped = true,
            "--deep" if !deep => deep = true,
            _ => return Err(format!("unknown or repeated MiMo gate option {option}").into()),
        }
    }
    if grouped && deep {
        return Err("--grouped and --deep cannot be combined".into());
    }
    let program = if deep {
        2
    } else if grouped {
        1
    } else {
        0
    };
    let config_bytes = std::fs::read(&args[0])?;
    if format!("{:x}", Sha256::digest(&config_bytes))
        != "61bea4a0f7a0dd8969f8cae528761e26b697dd12ff63e98804c3f0945492e621"
    {
        return Err("MiMo mixed gate requires pinned source config".into());
    }
    let gpu_index: usize = args[1].parse()?;
    let engine = Engine::new(gpu_index)?;
    println!("gpu_index\t{gpu_index}");
    println!("class\tseq\tmax_abs\twall_ms");
    if !attention_only {
        for width in [128, 192] {
            check_codec(&engine, width)?;
        }
    }
    for seq in [1, 127, 128, 129, 255, 256, 257, 512] {
        check_pattern(&engine, seq, attention_only, program)?;
    }
    check_constant(&engine, 32_769, program)?;
    if million {
        check_constant(&engine, 1_048_576, program)?;
    }
    if million_varied {
        check_varied_million(&engine, program)?;
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("mimo_mixed_attn_gate: {error}");
        std::process::exit(1);
    }
}
