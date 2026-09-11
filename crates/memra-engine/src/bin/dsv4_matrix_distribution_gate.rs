//! Teacher-forced reference/matrix distribution characterization, not quality admission.
//! Both programs consume the same real source tokens; neither feeds its own predictions.
use memra_engine::dsv4_gpu::{Dsv4Gpu, Dsv4SampleCfg, dsv4_sample_row};
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

const SCORED_ROWS: usize = 64;
const WINDOWS: [(usize, usize); 3] = [(0, 160), (32_768, 1025), (131_072, 4097)];

#[derive(Debug)]
struct RowMetrics {
    reference_nll: f64,
    matrix_nll: f64,
    kl_reference_matrix: f64,
    total_variation: f64,
    max_logit_delta: f64,
    reference_top1: usize,
    matrix_top1: usize,
}

fn log_probabilities(row: &[f32]) -> Result<Vec<f64>, &'static str> {
    if row.is_empty() || row.iter().any(|x| !x.is_finite()) {
        return Err("empty or non-finite logits");
    }
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let log_sum = row
        .iter()
        .map(|&x| (f64::from(x) - max).exp())
        .sum::<f64>()
        .ln();
    Ok(row
        .iter()
        .map(|&x| (f64::from(x) - max) - log_sum)
        .collect())
}

fn first_argmax(row: &[f32]) -> usize {
    (1..row.len()).fold(0, |best, i| if row[i] > row[best] { i } else { best })
}

fn compare_rows(a: &[f32], b: &[f32], target: usize) -> Result<RowMetrics, &'static str> {
    if a.len() != b.len() || target >= a.len() {
        return Err("logit shape or target mismatch");
    }
    let p = log_probabilities(a)?;
    let q = log_probabilities(b)?;
    let mut kl = 0.0;
    let mut tv = 0.0;
    for (&lp, &lq) in p.iter().zip(&q) {
        let pp = lp.exp();
        let pq = lq.exp();
        kl += pp * (lp - lq);
        tv += (pp - pq).abs();
    }
    Ok(RowMetrics {
        reference_nll: -p[target],
        matrix_nll: -q[target],
        kl_reference_matrix: kl,
        total_variation: tv * 0.5,
        max_logit_delta: a
            .iter()
            .zip(b)
            .map(|(&x, &y)| (f64::from(x) - f64::from(y)).abs())
            .fold(0.0, f64::max),
        reference_top1: first_argmax(a),
        matrix_top1: first_argmax(b),
    })
}

fn create_new(path: &Path) -> BufWriter<File> {
    BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .expect("create new receipt; never overwrite a previous run"),
    )
}

fn capture(gpu: &Dsv4Gpu, tokens: &[u32], prefix: usize, path: &Path) -> Vec<Vec<f32>> {
    assert_eq!(tokens.len(), prefix + SCORED_ROWS);
    let mut output = create_new(path);
    let mut state = gpu
        .alloc_decode_state_for_transient(tokens.len() + 32, 32)
        .expect("state");
    let mut draft = gpu.dspark_alloc_state().expect("draft");
    println!(
        "CAPTURE matrix={} prefix={prefix} rows={SCORED_ROWS}",
        gpu.matrix_moe_enabled()
    );
    let mut row = gpu
        .dspark_prefill_prime_chunked(&tokens[..prefix], &mut state, &mut draft, 32)
        .expect("prime");
    drop(draft);
    let mut bank = Vec::with_capacity(SCORED_ROWS);
    for step in 0..SCORED_ROWS {
        assert!(row.iter().all(|x| x.is_finite()));
        for value in &row {
            output.write_all(&value.to_le_bytes()).expect("bank logits");
        }
        bank.push(row);
        if step + 1 < SCORED_ROWS {
            row = gpu
                .decode_step(tokens[prefix + step], &mut state)
                .expect("forced source step");
        } else {
            break;
        }
    }
    output.flush().expect("flush logits");
    bank
}

fn main() {
    // Freeze this historical instrument independently of the newer defaults.
    // This is process startup, before any model or worker threads exist.
    unsafe {
        std::env::set_var("MEMRA_DSV4_HC_DOT_SPLIT", "0");
        std::env::set_var("MEMRA_DSV4_DENSE_FAST", "0");
        std::env::set_var("MEMRA_DSV4_NORM_FUSE", "0");
        std::env::set_var("MEMRA_DSV4_NORM_FUSE2", "0");
        std::env::set_var("MEMRA_DSV4_NORM2_WIDE", "0");
        // Gate-only AR phase instrument: pinned off here so no other bin can inherit
        // an exported instrument or null collective from the environment.
        std::env::set_var("MEMRA_DSV4_AR_PHASE", "0");
    }

    let args: Vec<String> = std::env::args().collect();
    assert_eq!(
        args.len(),
        4,
        "usage: dsv4_matrix_distribution_gate <model-dir> <real-source.txt> <new-out-dir>"
    );
    for (name, required) in [
        ("MEMRA_DSV4_DECODE_PATH", "device"),
        ("MEMRA_DSV4_EXPERT_ARM", "native"),
        ("MEMRA_DSV4_DENSE_ARM", "fp8"),
        ("MEMRA_DSV4_DRAFTER", "dspark"),
        ("MEMRA_DSV4_PREFILL_MOE", "reference"),
        ("MEMRA_DSV4_EP", "off"),
    ] {
        assert_eq!(
            std::env::var(name).as_deref(),
            Ok(required),
            "requires {name}={required}"
        );
    }
    let dir = Path::new(&args[1]);
    let output = Path::new(&args[3]);
    std::fs::create_dir(output).expect("create a new, nonexisting receipt directory");
    let text = std::fs::read_to_string(&args[2]).expect("source");
    let tokenizer = Tokenizer::from_hf_dir(dir).expect("tokenizer");
    let tokens = tokenizer.encode(&format!("Review this engine source:\n{text}"), true);
    println!(
        "SOURCE sha256={:x} tokens={} windows={WINDOWS:?} scored_rows_per_window={SCORED_ROWS}",
        Sha256::digest(text.as_bytes()),
        tokens.len()
    );
    assert!(
        WINDOWS
            .iter()
            .all(|&(offset, prefix)| offset + prefix + SCORED_ROWS <= tokens.len()),
        "source is too short; no padding or repeated tokens"
    );
    let capacity = WINDOWS.iter().map(|&(_, prefix)| prefix).max().unwrap() + SCORED_ROWS + 32;
    // The reference executor has no environment door any more (memra #461): this
    // gate loads it through the one arm there is, and then asserts it got it.
    memra_engine::arm_reference_expert_program_for_gate();
    let mut gpu =
        Dsv4Gpu::load(dir, &[0, 1], ActQuantVariant::RefFp8Round, capacity).expect("load");
    assert!(!gpu.matrix_moe_enabled(), "the gate arm must load reference");
    memra_engine::disarm_reference_expert_program_for_gate();
    gpu.set_grouped_route_device_for_gate(true)
        .expect("device routing");
    let cfg = Dsv4SampleCfg {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        seed: 20260906,
    };
    let mut metrics_file = create_new(&output.join("rows.jsonl"));
    for (window, &(offset, prefix)) in WINDOWS.iter().enumerate() {
        let window_tokens = &tokens[offset..offset + prefix + SCORED_ROWS];
        let mut token_file = create_new(&output.join(format!("window-{window}.tokens.u32le")));
        let mut token_hash = Sha256::new();
        for token in window_tokens {
            token_file.write_all(&token.to_le_bytes()).unwrap();
            token_hash.update(token.to_le_bytes());
        }
        token_file.flush().unwrap();
        println!(
            "WINDOW id={window} offset={offset} prefix={prefix} token_sha256={:x}",
            token_hash.finalize()
        );
        gpu.set_matrix_moe_for_gate(false)
            .expect("reference program");
        let before = gpu.grouped_device_route_calls();
        let reference = capture(
            &gpu,
            window_tokens,
            prefix,
            &output.join(format!("window-{window}.reference.f32le")),
        );
        assert_eq!(
            before,
            gpu.grouped_device_route_calls(),
            "reference must not engage grouped routes"
        );
        gpu.set_matrix_moe_for_gate(true).expect("matrix program");
        let matrix = capture(
            &gpu,
            window_tokens,
            prefix,
            &output.join(format!("window-{window}.matrix.f32le")),
        );
        assert!(
            gpu.grouped_device_route_calls() > before,
            "matrix device routes did not engage"
        );
        let mut sum = [0.0_f64; 4];
        let mut max_delta = 0.0_f64;
        let mut top1_agree = 0;
        let mut sampled_agree = 0;
        for step in 0..SCORED_ROWS {
            let a = &reference[step];
            let b = &matrix[step];
            let target = window_tokens[prefix + step];
            let m = compare_rows(a, b, target as usize).expect("valid distributions");
            let reference_sample = dsv4_sample_row(a, prefix + step, &cfg).unwrap();
            let matrix_sample = dsv4_sample_row(b, prefix + step, &cfg).unwrap();
            let line = format!(
                "{{\"window\":{window},\"offset\":{offset},\"prefix\":{prefix},\"step\":{step},\"vocab\":{},\"target\":{target},\"reference_nll\":{},\"matrix_nll\":{},\"kl_reference_matrix\":{},\"total_variation\":{},\"max_logit_delta\":{},\"reference_top1\":{},\"matrix_top1\":{},\"reference_sample\":{reference_sample},\"matrix_sample\":{matrix_sample},\"reference_sha256\":\"{:x}\",\"matrix_sha256\":\"{:x}\"}}",
                a.len(),
                m.reference_nll,
                m.matrix_nll,
                m.kl_reference_matrix,
                m.total_variation,
                m.max_logit_delta,
                m.reference_top1,
                m.matrix_top1,
                hash_row(a),
                hash_row(b)
            );
            writeln!(metrics_file, "{line}").unwrap();
            println!("ROW {line}");
            sum[0] += m.reference_nll;
            sum[1] += m.matrix_nll;
            sum[2] += m.kl_reference_matrix;
            sum[3] += m.total_variation;
            max_delta = max_delta.max(m.max_logit_delta);
            top1_agree += usize::from(m.reference_top1 == m.matrix_top1);
            sampled_agree += usize::from(reference_sample == matrix_sample);
        }
        metrics_file.flush().unwrap();
        println!(
            "SUMMARY window={window} rows={SCORED_ROWS} mean_reference_nll={} mean_matrix_nll={} mean_nll_delta={} mean_kl_reference_matrix={} mean_tv={} max_logit_delta={max_delta} top1_agree={top1_agree} sampled_agree={sampled_agree}",
            sum[0] / SCORED_ROWS as f64,
            sum[1] / SCORED_ROWS as f64,
            (sum[1] - sum[0]) / SCORED_ROWS as f64,
            sum[2] / SCORED_ROWS as f64,
            sum[3] / SCORED_ROWS as f64
        );
    }
    println!(
        "COMPLETE distribution characterization; no checkpoint quality, speed or serving admission inferred"
    );
}

fn hash_row(row: &[f32]) -> sha2::digest::Output<Sha256> {
    let mut hash = Sha256::new();
    for value in row {
        hash.update(value.to_le_bytes());
    }
    hash.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_and_shifted_logits_have_the_same_distribution() {
        let a = [1.0, 0.0, -2.0];
        for b in [a, [1001.0, 1000.0, 998.0]] {
            let m = compare_rows(&a, &b, 1).unwrap();
            assert_eq!(m.reference_nll, m.matrix_nll);
            assert_eq!(m.kl_reference_matrix, 0.0);
            assert_eq!(m.total_variation, 0.0);
        }
    }

    #[test]
    fn known_binary_distributions_match_analytic_metrics() {
        let m = compare_rows(&[0.0, 0.0], &[3.0_f32.ln(), 0.0], 1).unwrap();
        assert!((m.reference_nll - 2.0_f64.ln()).abs() < 1e-12);
        assert!((m.matrix_nll - 4.0_f64.ln()).abs() < 1e-7);
        assert!((m.kl_reference_matrix - (4.0_f64 / 3.0).ln() / 2.0).abs() < 1e-7);
        assert!((m.total_variation - 0.25).abs() < 1e-7);
        assert_eq!(m.reference_top1, 0);
        assert_eq!(m.matrix_top1, 0);
    }

    #[test]
    fn extreme_finite_logits_remain_finite() {
        let m = compare_rows(&[f32::MAX, -f32::MAX], &[-f32::MAX, f32::MAX], 1).unwrap();
        assert!(m.reference_nll.is_finite() && m.matrix_nll.is_finite());
        assert!(m.kl_reference_matrix.is_finite() && m.max_logit_delta.is_finite());
        assert_eq!(m.total_variation, 1.0);
    }

    #[test]
    fn invalid_rows_and_targets_are_refused() {
        for (a, b, target) in [
            (&[][..], &[][..], 0),
            (&[0.0][..], &[0.0, 1.0][..], 0),
            (&[0.0][..], &[0.0][..], 1),
            (&[f32::NAN][..], &[0.0][..], 0),
            (&[0.0][..], &[f32::INFINITY][..], 0),
        ] {
            assert!(compare_rows(a, b, target).is_err());
        }
    }
}
