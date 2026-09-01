//! Compare two raw f32 logit rows under the exact device Gumbel sampler.
//!
//! Usage: logit-sample-probe <a.f32> <b.f32> <seed> <stream-pos> <temp> <target-id>

use memra_engine::Engine;

fn read_f32(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "{path} has {} bytes, not a whole number of f32 values",
            bytes.len()
        )
        .into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn top_two(values: &[f32]) -> [(usize, f32); 2] {
    let mut top = [(usize::MAX, f32::NEG_INFINITY); 2];
    for (index, &value) in values.iter().enumerate() {
        if value > top[0].1 || (value == top[0].1 && index < top[0].0) {
            top[1] = top[0];
            top[0] = (index, value);
        } else if value > top[1].1 || (value == top[1].1 && index < top[1].0) {
            top[1] = (index, value);
        }
    }
    top
}

fn rank(values: &[f32], index: usize) -> usize {
    let value = values[index];
    1 + values
        .iter()
        .enumerate()
        .filter(|&(other_index, other)| *other > value || (*other == value && other_index < index))
        .count()
}

fn probability(values: &[f32], index: usize, temp: f32) -> f64 {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let inv_temp = 1.0 / temp as f64;
    let denominator: f64 = values
        .iter()
        .map(|&value| ((value as f64 - max) * inv_temp).exp())
        .sum();
    ((values[index] as f64 - max) * inv_temp).exp() / denominator
}

fn describe(
    label: &str,
    logits: &[f32],
    perturbed: &[f32],
    sampled: usize,
    target: usize,
    temp: f32,
) {
    let raw_top = top_two(logits);
    let sample_top = top_two(perturbed);
    println!(
        "{label} raw_argmax={} raw_margin={:.9} sampled={} sample_margin={:.9}",
        raw_top[0].0,
        raw_top[0].1 - raw_top[1].1,
        sampled,
        sample_top[0].1 - sample_top[1].1,
    );
    println!(
        "{label} target={} raw_logit={:.9} raw_rank={} probability_t{temp}={:.12e} gumbel_score={:.9} gumbel_rank={}",
        target,
        logits[target],
        rank(logits, target),
        probability(logits, target, temp),
        perturbed[target],
        rank(perturbed, target),
    );
    println!(
        "{label} sampled_raw_logit={:.9} sampled_probability_t{temp}={:.12e} sampled_gumbel_score={:.9}",
        logits[sampled],
        probability(logits, sampled, temp),
        perturbed[sampled],
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 7 {
        return Err(
            "usage: logit-sample-probe <a.f32> <b.f32> <seed> <stream-pos> <temp> <target-id>"
                .into(),
        );
    }
    let (a, b) = (read_f32(&args[1])?, read_f32(&args[2])?);
    if a.len() != b.len() || a.is_empty() {
        return Err(format!(
            "logit lengths differ or are empty: {} vs {}",
            a.len(),
            b.len()
        )
        .into());
    }
    let seed: u64 = args[3].parse()?;
    let stream_pos: u32 = args[4].parse()?;
    let temp: f32 = args[5].parse()?;
    let target: usize = args[6].parse()?;
    if target >= a.len() || temp <= 0.0 {
        return Err(format!(
            "target {target} or temperature {temp} is invalid for vocab {}",
            a.len()
        )
        .into());
    }

    let mut max_abs = (0usize, 0.0f32);
    let (mut sum_abs, mut sum_sq, mut ref_sq) = (0.0f64, 0.0f64, 0.0f64);
    for (index, (&av, &bv)) in a.iter().zip(&b).enumerate() {
        let delta = (av - bv).abs();
        if delta > max_abs.1 {
            max_abs = (index, delta);
        }
        sum_abs += delta as f64;
        sum_sq += (av as f64 - bv as f64).powi(2);
        ref_sq += (bv as f64).powi(2);
    }
    println!(
        "compare n={} max_abs={:.9} max_abs_id={} mean_abs={:.9} rms_abs={:.9} rms_rel={:.9}",
        a.len(),
        max_abs.1,
        max_abs.0,
        sum_abs / a.len() as f64,
        (sum_sq / a.len() as f64).sqrt(),
        (sum_sq / ref_sq.max(f64::MIN_POSITIVE)).sqrt(),
    );

    let engine = Engine::new(0)?;
    let mut sampled = Vec::with_capacity(2);
    let mut perturbed_rows = Vec::with_capacity(2);
    for logits in [&a, &b] {
        let device_logits = engine.htod(logits)?;
        let mut device_perturbed = engine.zeros(logits.len())?;
        engine.gumbel_perturb(
            &device_logits,
            &mut device_perturbed,
            logits.len(),
            seed,
            stream_pos,
            temp,
        )?;
        let token = engine.argmax_token_device(&device_perturbed, logits.len())?;
        sampled.push(engine.dtoh_u32(&token)?[0] as usize);
        perturbed_rows.push(engine.dtoh(&device_perturbed)?);
    }
    describe("a", &a, &perturbed_rows[0], sampled[0], target, temp);
    describe("b", &b, &perturbed_rows[1], sampled[1], target, temp);
    println!(
        "verdict same_raw_argmax={} same_sample={} target_selected_a={} target_selected_b={} target_logit_delta={:.9}",
        top_two(&a)[0].0 == top_two(&b)[0].0,
        sampled[0] == sampled[1],
        sampled[0] == target,
        sampled[1] == target,
        a[target] - b[target],
    );
    Ok(())
}
