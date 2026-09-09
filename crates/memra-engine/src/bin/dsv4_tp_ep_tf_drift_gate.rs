//! Report-only B/E teacher-forced incremental drift on the native TP2/EP path.
//! B and E consume the same pinned external token tape; only split-K changes.
//! B,E,B,E use fresh states in one load. Invariants and repeated-arm identity
//! assert; incremental drift and historical CPU compatibility never certify quality.
//! Usage: dsv4_tp_ep_tf_drift_gate <model> <tf.json> <source-bank.json> <out-dir>

use memra_engine::dsv4_gpu::Dsv4Gpu;
use memra_engine::dsv4_sampler::{Dsv4Sampler, dsv4_sampler};
use memra_gguf::config::JsonObj;
use memra_gguf::dsv4_forward::{ActQuantVariant, drift_coeff};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};

const TF_SHA: &str = "63a6e13500b3a3b843595f2b6f55e27d55ea299006a6c6ec085d3b845847946b";
const SOURCE_SHA: &str = "920901870d3a0925d595c3d1f0190f5c9aa443426d074455a02660efbc3eb030";
const VOCAB: usize = 129280;
const TOP_K: usize = 20;

#[derive(Clone, Copy, Debug, Default)]
struct Counters {
    rank_layer: [u64; 2],
    ep: u64,
    ar: u64,
    attention_rank: [u64; 2],
    attention_ar: u64,
    gu_m1: u64,
    splitk_gu: u64,
    splitk_down: u64,
    gu_half2: u64,
    down_half2: u64,
    wo_a: u64,
    index_radix: u64,
    small_launches: [u64; 2],
}

fn counters(gpu: &Dsv4Gpu) -> Counters {
    Counters {
        rank_layer: gpu.tp_ep_rank_layer_calls(),
        ep: gpu.ep_calls(),
        ar: gpu.tp_ep_ar_dispatches(),
        attention_rank: gpu.attention_tp_rank_calls(),
        attention_ar: gpu.attention_tp_ar_calls(),
        splitk_gu: memra_engine::MOE_M1_SPLITK_GU_DISPATCHES
            .load(std::sync::atomic::Ordering::Relaxed),
        splitk_down: memra_engine::MOE_M1_SPLITK_DOWN_DISPATCHES
            .load(std::sync::atomic::Ordering::Relaxed),
        gu_m1: memra_engine::moe_f16g_gu_m1_tc_dispatches(),
        gu_half2: memra_engine::moe_f16g_gu_half2_dispatches(),
        down_half2: memra_engine::moe_f16g_down_m1_half2_dispatches(),
        wo_a: gpu.dense_wo_a_grouped_dispatches(),
        index_radix: gpu.index_topk_radix_dispatches(),
        small_launches: gpu.small_kernel_launches(),
    }
}

fn delta(after: Counters, before: Counters) -> Counters {
    Counters {
        rank_layer: [
            after.rank_layer[0] - before.rank_layer[0],
            after.rank_layer[1] - before.rank_layer[1],
        ],
        ep: after.ep - before.ep,
        ar: after.ar - before.ar,
        attention_rank: std::array::from_fn(|rank| {
            after.attention_rank[rank] - before.attention_rank[rank]
        }),
        attention_ar: after.attention_ar - before.attention_ar,
        splitk_gu: after.splitk_gu - before.splitk_gu,
        splitk_down: after.splitk_down - before.splitk_down,
        gu_m1: after.gu_m1 - before.gu_m1,
        gu_half2: after.gu_half2 - before.gu_half2,
        down_half2: after.down_half2 - before.down_half2,
        wo_a: after.wo_a - before.wo_a,
        index_radix: after.index_radix - before.index_radix,
        small_launches: std::array::from_fn(|i| after.small_launches[i] - before.small_launches[i]),
    }
}

fn assert_engagement(gpu: &Dsv4Gpu, c: Counters, prime: usize, decode: usize) {
    let layers = gpu.topology().layers as u64;
    let attention_mode = gpu.attention_tp_geometry().is_some();
    let attention_steps = u64::from(attention_mode) * (prime + decode) as u64 * layers;
    for (rank, &calls) in c.rank_layer.iter().enumerate() {
        assert_eq!(
            calls,
            (prime + decode) as u64 * layers,
            "rank {rank} layer-walk engagement"
        );
    }
    assert_eq!(
        c.ep,
        2 * (prime + decode) as u64 * layers,
        "local EP engagement"
    );
    assert_eq!(
        c.ar,
        (prime + decode) as u64 * layers + attention_steps,
        "one-shot AR engagement"
    );
    assert_eq!(
        c.attention_rank, [attention_steps; 2],
        "actual attention rank producers"
    );
    assert_eq!(
        c.attention_ar, attention_steps,
        "actual attention reductions"
    );
    let local_steps = 2 * (prime + decode) as u64 * layers;
    let expected_small = if gpu.small_kernel_diet_enabled() {
        [2, 1]
    } else {
        [6, 2]
    };
    assert_eq!(
        c.small_launches,
        expected_small.map(|n| n * local_steps),
        "HC and Q pack actual enqueues: a diet PASS with the old count is forbidden"
    );
    let splitk = memra_engine::moe_m1_splitk_on();
    let oracle_steps = if splitk { 0 } else { local_steps };
    let splitk_steps = if splitk { local_steps } else { 0 };
    assert_eq!(c.gu_m1, oracle_steps, "GU-M1 actual enqueues");
    assert_eq!(c.gu_half2, oracle_steps, "GU-half2 actual enqueues");
    assert_eq!(c.down_half2, oracle_steps, "down-half2 actual enqueues");
    assert_eq!(c.splitk_gu, splitk_steps, "split-K GU two-pass enqueues");
    assert_eq!(
        c.splitk_down, splitk_steps,
        "split-K down two-pass enqueues"
    );
    println!(
        "MOE_ENGAGEMENT splitk={splitk} gu={} down={}",
        c.splitk_gu, c.splitk_down
    );
    assert_eq!(
        c.wo_a,
        if attention_mode { 0 } else { local_steps },
        "attention TP uses per-group wo_a; replicated attention uses qualified grouped wo_a"
    );
    // At prompt+output <= 512, the radix selector's N=2048 eligibility is
    // intentionally inert. The arm is still reported and checked as zero.
    assert_eq!(
        c.index_radix, 0,
        "short-context radix selector must be inert"
    );
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn row_digest(row: &[f32]) -> String {
    let mut h = Sha256::new();
    for x in row {
        h.update(x.to_bits().to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
fn drain(gpu: &Dsv4Gpu) {
    for stage in &gpu.stages {
        stage.gpu.stream().synchronize().expect("drain ranks");
    }
}
fn finite(row: &[f32]) {
    assert!(
        !row.is_empty() && row.iter().all(|x| x.is_finite()),
        "finite nonempty logits"
    );
}
fn top(row: &[f32], k: usize) -> Vec<usize> {
    finite(row);
    assert!(k > 0 && k < row.len());
    let mut ids: Vec<_> = (0..row.len()).collect();
    let cmp = |a: &usize, b: &usize| row[*b].partial_cmp(&row[*a]).unwrap().then(a.cmp(b));
    ids.select_nth_unstable_by(k, cmp);
    ids.truncate(k);
    ids.sort_unstable_by(cmp);
    ids
}
// Neumaier summation also handles the signed terms in KL.
fn sum(values: impl Iterator<Item = f64>) -> f64 {
    let (mut total, mut correction) = (0.0f64, 0.0f64);
    for x in values {
        let next = total + x;
        correction += if total.abs() >= x.abs() {
            (total - next) + x
        } else {
            (x - next) + total
        };
        total = next;
    }
    total + correction
}
fn log_probs(row: &[f32]) -> Vec<f64> {
    finite(row);
    let max = row
        .iter()
        .map(|&x| f64::from(x))
        .fold(f64::NEG_INFINITY, f64::max);
    let log_sum = sum(row.iter().map(|&x| (f64::from(x) - max).exp())).ln();
    row.iter()
        .map(|&x| (f64::from(x) - max) - log_sum)
        .collect()
}
#[derive(Debug)]
struct Metrics {
    max_abs: f64,
    max_abs_id: usize,
    rms: f64,
    relative_l2: Option<f64>,
    kl_b_e: f64,
    kl_e_b: f64,
    bank_logprob_delta: f64,
}
fn metrics(b: &[f32], e: &[f32], bank: usize) -> Metrics {
    finite(b);
    finite(e);
    assert_eq!(b.len(), e.len());
    assert!(bank < b.len());
    let diffs: Vec<_> = b
        .iter()
        .zip(e)
        .map(|(&b, &e)| f64::from(e) - f64::from(b))
        .collect();
    let norm = sum(b.iter().map(|&b| f64::from(b).powi(2)));
    let error = sum(diffs.iter().map(|x| x * x));
    let max_abs_id = diffs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()).then(b.0.cmp(&a.0)))
        .unwrap()
        .0;
    let pb = log_probs(b);
    let pe = log_probs(e);
    Metrics {
        max_abs: diffs[max_abs_id].abs(),
        max_abs_id,
        rms: (error / b.len() as f64).sqrt(),
        relative_l2: (norm > 0.0).then(|| (error / norm).sqrt()),
        kl_b_e: sum(pb.iter().zip(&pe).map(|(&b, &e)| b.exp() * (b - e))),
        kl_e_b: sum(pe.iter().zip(&pb).map(|(&e, &b)| e.exp() * (e - b))),
        bank_logprob_delta: pe[bank] - pb[bank],
    }
}
fn historical(pick: usize, ids: &[u32], logits: &[f32]) -> (&'static str, f64, f64) {
    let band = 3.0
        * 2f64.sqrt()
        * (drift_coeff(86.0, 86.0) + drift_coeff(0.0, 86.0))
        * f64::from(logits[0]).abs();
    if pick == ids[0] as usize {
        return ("agree", 0.0, band);
    }
    match ids.iter().position(|&id| id as usize == pick) {
        Some(k) => {
            let margin = f64::from(logits[0] - logits[k]);
            (
                if margin <= band {
                    "in_band"
                } else {
                    "out_of_band"
                },
                margin,
                band,
            )
        }
        None => {
            let lower = f64::from(logits[0] - logits[7]);
            (
                if lower > band {
                    "out_of_band"
                } else {
                    "unresolved"
                },
                lower,
                band,
            )
        }
    }
}
fn paired_line(pair: usize, position: usize, bank: u32, b: &[f32], e: &[f32]) -> String {
    let m = metrics(b, e, bank as usize);
    let bt = top(b, TOP_K);
    let et = top(e, TOP_K);
    let overlap: Vec<_> = [1, 5, 8, 20]
        .iter()
        .map(|&k| bt[..k].iter().filter(|id| et[..k].contains(id)).count())
        .collect();
    let rel = m
        .relative_l2
        .map_or_else(|| "null".to_owned(), |x| format!("{x:.17e}"));
    format!(
        concat!(
            "{{\"pair\":{},\"offset\":{},\"absolute_position\":{},\"bank_token\":{},",
            "\"max_abs\":{:.17e},\"max_abs_id\":{},\"rms\":{:.17e},\"relative_l2_B_denominator\":{},",
            "\"B_top20\":{:?},\"E_top20\":{:?},\"top_1_5_8_20_overlap\":{:?},",
            "\"B_top1_margin\":{:.17e},\"E_top1_margin\":{:.17e},",
            "\"B_margin_Bpick_minus_Epick\":{:.17e},\"E_margin_Epick_minus_Bpick\":{:.17e},",
            "\"bank_logprob_E_minus_B\":{:.17e},\"KL_B_E_nats\":{:.17e},\"KL_E_B_nats\":{:.17e},",
            "\"B_logits_sha256\":\"{}\",\"E_logits_sha256\":\"{}\",\"quality_verdict\":\"not_assessed\"}}"
        ),
        pair,
        position,
        32 + position,
        bank,
        m.max_abs,
        m.max_abs_id,
        m.rms,
        rel,
        bt,
        et,
        overlap,
        f64::from(b[bt[0]]) - f64::from(b[bt[1]]),
        f64::from(e[et[0]]) - f64::from(e[et[1]]),
        f64::from(b[bt[0]]) - f64::from(b[et[0]]),
        f64::from(e[et[0]]) - f64::from(e[bt[0]]),
        m.bank_logprob_delta,
        m.kl_b_e,
        m.kl_e_b,
        row_digest(b),
        row_digest(e)
    )
}
fn main() {
    // Freeze this historical instrument independently of the newer defaults.
    // This is process startup, before any model or worker threads exist.
    unsafe {
        std::env::set_var("MEMRA_DSV4_DENSE_FAST", "0");
        std::env::set_var("MEMRA_DSV4_NORM_FUSE", "0");
    }

    let args: Vec<_> = std::env::args().collect();
    assert_eq!(
        args.len(),
        5,
        "usage: dsv4_tp_ep_tf_drift_gate <model> <tf.json> <source-bank.json> <out-dir>"
    );
    let tf_bytes = fs::read(&args[2]).expect("teacher bank");
    let source_bytes = fs::read(&args[3]).expect("source bank");
    assert_eq!(digest(&tf_bytes), TF_SHA, "immutable teacher bank");
    assert_eq!(digest(&source_bytes), SOURCE_SHA, "immutable source bank");
    let tf = JsonObj::parse(std::str::from_utf8(&tf_bytes).unwrap());
    let source = JsonObj::parse(std::str::from_utf8(&source_bytes).unwrap());
    assert_eq!(tf.string("source_sha256").as_deref(), Some(SOURCE_SHA));
    assert_eq!(tf.string("variant").as_deref(), Some("ref"));
    assert_eq!(source.string("variant").as_deref(), Some("ref"));
    let prompt = tf.u32_array("prompt").unwrap();
    let tokens = tf.u32_array("tokens").unwrap();
    assert_eq!(prompt, source.u32_array("prompt").unwrap());
    assert_eq!(tokens, source.u32_array("tokens_run0").unwrap());
    assert_eq!(prompt.len(), 32);
    assert_eq!(tokens.len(), 160);
    assert!(prompt.iter().chain(&tokens).all(|&x| (x as usize) < VOCAB));
    let ids = tf.u32_array("top8_ids").unwrap();
    let logits = tf.f32_array("top8_logits").unwrap();
    assert_eq!(ids.len(), 160 * 8);
    assert_eq!(logits.len(), 160 * 8);
    finite(&logits);
    assert!(ids.iter().all(|&x| (x as usize) < VOCAB));
    for (i, &token) in tokens.iter().enumerate() {
        assert_eq!(ids[8 * i], token);
        assert!(logits[8 * i..8 * i + 8].windows(2).all(|w| w[0] >= w[1]));
    }
    for (name, value) in [
        ("MEMRA_DSV4_DECODE_PATH", "device"),
        ("MEMRA_DSV4_EXPERT_ARM", "native"),
        ("MEMRA_DSV4_DENSE_ARM", "fp8"),
        ("MEMRA_DSV4_DOTS_ARM", "f32x"),
        ("MEMRA_DSV4_EP", "pair"),
        ("MEMRA_DSV4_MOE_PROGRAM", "matrix"),
        ("MEMRA_DSV4_GROUPED_ROUTE", "device"),
        ("MEMRA_DSV4_VERIFY_TOPK", "device"),
        ("MEMRA_DSV4_PREFILL_MOE", "reference"),
        ("MEMRA_DSV4_DRAFTER", "off"),
        ("MEMRA_DSV4_SMALL_KERNEL_DIET", "1"),
        ("MEMRA_MOE_F16G", "2"),
        ("MEMRA_F16G_SK", "32"),
    ] {
        assert_eq!(
            std::env::var(name).as_deref(),
            Ok(value),
            "requires {name}={value}"
        );
    }
    assert_eq!(dsv4_sampler().expect("sampler door"), Dsv4Sampler::Device);
    fs::create_dir(&args[4]).expect("new output directory; never overwrite a receipt");
    let out = Path::new(&args[4]);
    let mut paired = BufWriter::new(fs::File::create(out.join("paired.jsonl")).unwrap());
    let mut positions = BufWriter::new(fs::File::create(out.join("positions.jsonl")).unwrap());
    let start = Instant::now();
    Dsv4Gpu::set_tp_ep_topology_for_gate(true);
    Dsv4Gpu::set_attention_tp_for_gate(true);
    // Pin the historical control program independently of the graph default.
    memra_engine::set_moe_m1_graph_splitk_for_gate(false);
    memra_engine::set_moe_m1_splitk_for_gate(false);
    let gpu = Dsv4Gpu::load(
        Path::new(&args[1]),
        &[0, 1],
        ActQuantVariant::RefFp8Round,
        224,
    )
    .expect("native model");
    assert!(gpu.topology().is_tp_ep());
    assert_eq!(gpu.topology().layers, 43);
    assert!(gpu.attention_tp_geometry().is_some() && gpu.small_kernel_diet_enabled());
    gpu.set_grouped_route_validation_for_gate(false);
    gpu.set_grouped_mirror_validation_for_gate(false);
    gpu.set_grouped_gu_fuse_for_gate(true);
    gpu.set_grouped_m1_tc_for_gate(true);
    memra_engine::set_moe_f16g_gu_m1_tc_for_gate(true);
    memra_engine::set_moe_f16g_gu_half2_for_gate(true);
    memra_engine::set_moe_f16g_down_m1_half2_for_gate(true);
    gpu.set_dense_wo_a_grouped_for_gate(false);
    gpu.set_index_topk_radix_for_gate(true);
    println!(
        "PROTOCOL arms=B,E,B,E prompt=32 positions=160 fed_tokens=external_bank fresh_state=true one_load=true sampler_draws=0 incremental=report_only historical_cpu_rule=report_only load_seconds={:.3}",
        start.elapsed().as_secs_f64()
    );
    let mut refs: [Option<Vec<Vec<f32>>>; 2] = [None, None];
    let mut final_refs = [None, None];
    for run in 0..4 {
        let e = run % 2 == 1;
        let arm = if e { "E" } else { "B" };
        let index = usize::from(e);
        gpu.set_grouped_m1_splitk_for_gate(e);
        println!(
            "ARM run={run} arm={arm} splitk={e} numeric_class={}",
            if e {
                memra_engine::MOE_M1_SPLITK_NUMERIC_CLASS
            } else {
                "existing_m1_f16_mma"
            }
        );
        let mut state = gpu
            .alloc_decode_state_for_transient(224, 1)
            .expect("fresh state");
        let before = counters(&gpu);
        let mut row = gpu
            .prefill_with_cache_chunked(&prompt[..1], &mut state, 1)
            .expect("single-token prime");
        for &token in &prompt[1..] {
            row = gpu.decode_step(token, &mut state).expect("banked prompt");
        }
        drain(&gpu);
        assert_eq!(state.pos, 32);
        let after_prime = counters(&gpu);
        assert_engagement(&gpu, delta(after_prime, before), 32, 0);
        let mut rows = Vec::new();
        let mut h = Sha256::new();
        let mut historical_counts = [0usize; 4];
        for (i, &bank) in tokens.iter().enumerate() {
            assert_eq!(row.len(), VOCAB);
            finite(&row);
            assert_eq!(state.pos, 32 + i);
            assert_eq!(gpu.tp_ep_ar_refusal_words().expect("refusal read"), [0, 0]);
            if let Some(reference) = &refs[index] {
                assert!(
                    row.iter()
                        .zip(&reference[i])
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "within-arm logits differ: run={run} arm={arm} position={i}"
                );
            }
            for x in &row {
                h.update(x.to_bits().to_le_bytes());
            }
            let picked = top(&row, TOP_K)[0];
            let (category, margin, band) =
                historical(picked, &ids[i * 8..i * 8 + 8], &logits[i * 8..i * 8 + 8]);
            historical_counts[match category {
                "agree" => 0,
                "in_band" => 1,
                "out_of_band" => 2,
                _ => 3,
            }] += 1;
            let line = format!(
                "{{\"run\":{run},\"arm\":\"{arm}\",\"offset\":{i},\"absolute_position\":{},\"bank_token\":{bank},\"pick\":{picked},\"logits_sha256\":\"{}\",\"historical_cpu_category\":\"{category}\",\"historical_cpu_margin_or_lower_bound\":{margin:.17e},\"historical_cpu_band\":{band:.17e},\"quality_verdict\":\"not_assessed\"}}",
                state.pos,
                row_digest(&row)
            );
            writeln!(positions, "{line}").unwrap();
            println!("POSITION {line}");
            if e {
                let line = paired_line(run / 2, i, bank, &refs[0].as_ref().unwrap()[i], &row);
                writeln!(paired, "{line}").unwrap();
                println!("PAIR {line}");
            }
            if refs[index].is_none() {
                rows.push(row.clone());
            }
            if i + 1 < tokens.len() {
                row = gpu
                    .decode_step(bank, &mut state)
                    .expect("externally teacher-forced step");
            }
        }
        drain(&gpu);
        assert_eq!(state.pos, 191);
        let after_decode = counters(&gpu);
        let prime = delta(after_prime, before);
        let decode = delta(after_decode, after_prime);
        assert_engagement(&gpu, decode, 0, 159);
        assert_eq!(gpu.tp_ep_ar_refusal_words().unwrap(), [0, 0]);
        let cache = gpu.tp_ep_cache_digest_for_gate(&state).unwrap();
        let hidden = gpu.tp_ep_hidden_digest_for_gate(&state).unwrap();
        assert_eq!(cache[0], cache[1]);
        assert_eq!(hidden[0], hidden[1]);
        let join = gpu.attention_tp_last_join_for_gate(&state).unwrap();
        for (i, (&a, &b)) in join.partials[0].iter().zip(&join.partials[1]).enumerate() {
            let expected = a + b;
            assert!(expected.is_finite());
            for joined in &join.joined {
                assert_eq!(
                    joined[i].to_bits(),
                    expected.to_bits(),
                    "rank-ordered f32 join"
                );
            }
        }
        let stream_sha = format!("{:x}", h.finalize());
        let end = (cache, hidden, row_digest(&join.joined[0]));
        if let Some(reference) = &final_refs[index] {
            assert_eq!(&end, reference, "within-arm final state");
        } else {
            final_refs[index] = Some(end);
        }
        if refs[index].is_none() {
            refs[index] = Some(rows);
        }
        println!(
            "RUN run={run} arm={arm} positions=160 final_pos=191 logits_sha256={stream_sha} historical_agree_inband_outofband_unresolved={historical_counts:?} prime_counters={prime:?} decode_counters={decode:?} invariants_pass=true quality_verdict=not_assessed"
        );
        positions.flush().unwrap();
        paired.flush().unwrap();
    }
    gpu.set_grouped_m1_splitk_for_gate(false);
    println!(
        "COMPLETE runs=4 paired_rows=320 repeated_logits_identical=true invariants_pass=true incremental_drift=report_only historical_cpu_compatibility=report_only quality_verdict=not_assessed performance_claim=false elapsed_seconds={:.3}",
        start.elapsed().as_secs_f64()
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stable_log_probabilities_and_shift_invariant_kl() {
        let b = [10000.0, 10001.0, 9999.0];
        let e = [20000.0, 20001.0, 19999.0];
        let m = metrics(&b, &e, 1);
        assert_eq!(m.kl_b_e, 0.0);
        assert_eq!(m.bank_logprob_delta, 0.0);
        assert!((sum(log_probs(&b).iter().map(|x| x.exp())) - 1.0).abs() < 1e-15);
    }
    #[test]
    fn asymmetric_kl_and_zero_denominator_are_reported() {
        let b = [0.0, 0.0];
        let e = [3.0f32.ln(), 0.0];
        let m = metrics(&b, &e, 0);
        assert!(m.relative_l2.is_none());
        assert!((m.kl_b_e - 0.5 * (4.0f64 / 3.0).ln()).abs() < 1e-7);
        assert!((m.kl_e_b - (0.75 * 1.5f64.ln() + 0.25 * 0.5f64.ln())).abs() < 1e-7);
        assert!((m.bank_logprob_delta - 1.5f64.ln()).abs() < 1e-7);
    }
    #[test]
    fn finite_ties_choose_lowest_id_including_signed_zero() {
        assert_eq!(top(&[-0.0, 0.0, -1.0], 2), vec![0, 1]);
        assert!(std::panic::catch_unwind(|| log_probs(&[f32::NAN])).is_err());
    }
}
