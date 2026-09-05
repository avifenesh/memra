//! Corrected grouped-prefill bring-up: complete MoE, FP8-QAT transport, and
//! same-forced-path logit characterization. No production/performance promotion.
use memra_engine::dsv4_gpu::{Dsv4Gpu, Dsv4SampleCfg, dsv4_sample_row};
use memra_gguf::dsv4_forward::FixtureSpec;
use std::path::Path;

fn compose(classes: &[(String, Vec<f32>)]) {
    let get = |name| {
        &classes
            .iter()
            .find(|(key, _)| key == name)
            .expect("class")
            .1
    };
    let (routed, shared, total) = (get("routed"), get("shared"), get("total"));
    assert_eq!(routed.len(), shared.len());
    assert_eq!(total.len(), shared.len());
    assert!(
        shared.iter().any(|value| value.abs() > 1e-8),
        "shared expert was not computed"
    );
    let mut changes = 0;
    for ((r, s), y) in routed.iter().zip(shared).zip(total) {
        assert!(r.is_finite() && s.is_finite() && y.is_finite());
        assert_eq!(
            (r + s).to_bits(),
            y.to_bits(),
            "total must include routed AND shared"
        );
        changes += usize::from(r.to_bits() != y.to_bits());
    }
    assert!(changes > 0, "shared expert has no effect");
}

fn compare_rows(reference: &[f32], candidate: &[f32]) -> (f32, f64) {
    assert_eq!(reference.len(), candidate.len());
    assert!(!reference.is_empty());
    let probs = |row: &[f32]| {
        assert!(row.iter().all(|value| value.is_finite()));
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut p: Vec<f64> = row
            .iter()
            .map(|&value| ((value - max) as f64).exp())
            .collect();
        let sum: f64 = p.iter().sum();
        for value in &mut p {
            *value /= sum;
        }
        p
    };
    let p = probs(reference);
    let q = probs(candidate);
    let tv = p.iter().zip(q).map(|(p, q)| (p - q).abs()).sum::<f64>() * 0.5;
    let max = reference
        .iter()
        .zip(candidate)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    (max, tv)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(
        args.len(),
        3,
        "usage: dsv4_grouped_prefill_gate <model-dir> <fixtures>"
    );
    let fixture = FixtureSpec::load(Path::new(&args[2]));
    let prompt = fixture.tokens_160.as_ref().expect("160-token real fixture");
    let mut gpu = Dsv4Gpu::load(Path::new(&args[1]), &[0, 1], fixture.variant, 4096).expect("load");
    for layer in [3, 42] {
        for rows in [1, 32, 64] {
            let input: Vec<f32> = (0..rows * 4096)
                .map(|i| ((i * 97 % 1009) as f32 - 504.0) / 503.0)
                .collect();
            gpu.set_prefill_grouped_for_gate(false).expect("reference");
            let reference = gpu
                .moe_components_for_gate(layer, &prompt[..rows], &input)
                .expect("reference MoE");
            compose(&reference);
            gpu.set_prefill_grouped_for_gate(true).expect("grouped");
            let grouped = gpu
                .moe_components_for_gate(layer, &prompt[..rows], &input)
                .expect("grouped MoE");
            compose(&grouped);
            assert_eq!(reference[1].1, grouped[1].1, "shared path changed");
            let max = reference[2]
                .1
                .iter()
                .zip(&grouped[2].1)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            println!(
                "COMPOSE layer={layer} rows={rows} shared identical; total=routed+shared max_total_delta={max}"
            );
        }
    }
    let sample = Dsv4SampleCfg {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        seed: 20260905,
    };
    for width in [32, 64] {
        gpu.set_prefill_grouped_for_gate(false).expect("reference");
        let mut reference_state = gpu
            .alloc_decode_state_for_transient(192, width)
            .expect("reference state");
        let mut reference = gpu
            .prefill_with_cache_chunked(prompt, &mut reference_state, width)
            .expect("reference prefill");
        gpu.set_prefill_grouped_for_gate(true).expect("grouped");
        let mut grouped_state = gpu
            .alloc_decode_state_for_transient(192, width)
            .expect("grouped state");
        let mut grouped = gpu
            .prefill_with_cache_chunked(prompt, &mut grouped_state, width)
            .expect("grouped prefill");
        let mut mismatches = 0;
        for step in 0..16 {
            let (max, tv) = compare_rows(&reference, &grouped);
            let token = dsv4_sample_row(&reference, prompt.len() + step, &sample)
                .expect("reference sample");
            let candidate =
                dsv4_sample_row(&grouped, prompt.len() + step, &sample).expect("grouped sample");
            mismatches += usize::from(token != candidate);
            println!(
                "CHARACTERIZE width={width} step={step} max_logit_delta={max} tv={tv:.9} sampled_match={}",
                token == candidate
            );
            reference = gpu
                .decode_step(token, &mut reference_state)
                .expect("forced reference");
            grouped = gpu
                .decode_step(token, &mut grouped_state)
                .expect("forced grouped");
        }
        println!(
            "CHARACTERIZE width={width} sampled_mismatches={mismatches}/16; not an admission threshold"
        );
    }
    println!(
        "PASS complete MoE composition and finite teacher-forcing characterization; serving qualification not claimed"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[should_panic(expected = "total must include routed AND shared")]
    fn omitted_shared_is_rejected() {
        compose(&[
            ("routed".into(), vec![1.0]),
            ("shared".into(), vec![2.0]),
            ("total".into(), vec![1.0]),
        ]);
    }
    #[test]
    #[should_panic(expected = "shared expert was not computed")]
    fn uncomputed_shared_is_rejected() {
        compose(&[
            ("routed".into(), vec![1.0]),
            ("shared".into(), vec![0.0]),
            ("total".into(), vec![1.0]),
        ]);
    }
}
