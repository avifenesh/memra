//! DSV4 dense-path NUMERIC CLASS determination: scalar GEMV against CUTLASS split-K.
//!
//! The question this binary answers is not "is cutlass faster" (measured elsewhere:
//! 1.386x at width 512 on the engine prefill, darklanes #708) and not "is cutlass
//! good enough" (an owner decision). It answers exactly one thing: how far the two
//! dense paths drift, in the darklanes #534 shape, so the default verdict has rows
//! instead of adjectives. Bit-equality is unreachable by construction (mma fragment
//! registers per 128-k block against the shipped 128-leaf smem halving tree), so
//! this gate prints drift, never identity.
//!
//! The determination is made on evidence, in this order, and every step is a
//! refusal if it does not hold:
//!
//! 1. WITHIN-ARM IDENTITY FIRST. Panels run A,B,B,A from one load. The second A
//!    must be bit-identical to the first, and the second B to the first. A program
//!    that is not deterministic against itself cannot be compared to another one,
//!    so a repeat mismatch aborts before any cross-arm number is printed. This is
//!    also the gate's identity arm: it is the case where the comparison MUST report
//!    zero drift, and it runs on real rows, not synthetic ones.
//! 2. ENGAGEMENT. The scalar arm must run zero split-K calls; the cutlass arm must
//!    run nonzero ones AND cross a device boundary (ws_device_flips > 0), because
//!    the defect this path exists to fix only bites across cards. Two arms that did
//!    not actually run two programs produce a PASS that lies.
//! 3. CROSS-ARM ROWS in the darklanes #534 shape: per-row bit equality, top-1 pick
//!    for each arm, KL in BOTH directions, total variation, target NLL, and the
//!    seeded sampled draw. Aggregates carry KL mean AND max per direction, because
//!    #534's finding was that a changed f32 tree flips near-tie argmax, which a
//!    top-1 count catches and an aggregate KL hides.
//!
//! PRODUCT SEMANTICS ARE CHECKED SEPARATELY FROM DRIFT, because they are not
//! drift: a change that perturbs them is wrong at any speed. The sampler RNG
//! stream is position-keyed (`dsv4_pos_uniform(seed, pos)`, one draw per row, pure
//! in its two arguments). The gate banks the uniform for every scored position
//! under both arms and refuses on the first difference. The drawn TOKEN may still
//! differ, because the CDF the stream indexes moved; that is a drift consequence
//! and it is reported as one, never as a stream change. (The expert-program
//! resume-refusal battery does not apply to this axis: the dense dispatch moves no
//! refusal.)
//!
//! NON-VACUITY runs before the GPU is touched: a synthetic pair with one flipped
//! bit must move the bit-equality, max-delta, KL and top-1 counters. A comparison
//! that cannot see a planted difference cannot certify the absence of one.
//!
//! This gate does NOT admit the cutlass path for quality, and prints no threshold.
//! ROW FIELD NAMES. `scalar_*` and `cutlass_*` are the DENSE axis's names; every
//! suffix after the prefix matches the #534 shape so one comparator reads every
//! run. Every row carries `"axis":"dense_cutlass"` so a receipt says which.
//!
//! Usage: dsv4_dense_class_gate <model-dir> <panels.txt> <panels-sha256> <new-out-dir>
use memra_engine::dsv4_gpu::{
    Dsv4Gpu, Dsv4SampleCfg, arm_dense_cutlass_for_gate, dense_cutlass_armed_for_gate,
    dense_cutlass_counts_for_gate, dsv4_pos_uniform, dsv4_sample_row,
};
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

/// Scored rows per panel. Every scored row is teacher-forced from the panel's own
/// real tokens; neither program ever consumes its own prediction, so the arms never
/// diverge into different histories and the comparison stays paired.
const SCORED_ROWS: usize = 64;
const PRIME_CHUNK: usize = 32;
const SEED: u64 = 20260910;

#[derive(Clone, Copy, Debug)]
struct Row {
    bits_equal: bool,
    max_abs: f64,
    max_abs_id: usize,
    kl_ab: f64,
    kl_ba: f64,
    tv: f64,
    nll_a: f64,
    nll_b: f64,
    top1_a: usize,
    top1_b: usize,
}

fn log_probabilities(row: &[f32]) -> Result<Vec<f64>, &'static str> {
    if row.is_empty() || row.iter().any(|x| !x.is_finite()) {
        return Err("empty or non-finite logits");
    }
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let log_sum = neumaier(row.iter().map(|&x| (f64::from(x) - max).exp())).ln();
    Ok(row
        .iter()
        .map(|&x| (f64::from(x) - max) - log_sum)
        .collect())
}

/// Neumaier summation: KL carries signed terms that cancel, and a naive sum over
/// 129k of them loses exactly the small differences this gate exists to measure.
fn neumaier(values: impl Iterator<Item = f64>) -> f64 {
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

/// House tie ordering: value descending, index ascending. `f32::max`-style folding
/// would pick the LAST maximum on a tie; the served candidate order picks the first.
fn first_argmax(row: &[f32]) -> usize {
    (1..row.len()).fold(0, |best, i| if row[i] > row[best] { i } else { best })
}

fn compare(a: &[f32], b: &[f32], target: usize) -> Result<Row, &'static str> {
    if a.len() != b.len() || target >= a.len() {
        return Err("logit shape or target mismatch");
    }
    let p = log_probabilities(a)?;
    let q = log_probabilities(b)?;
    let (mut max_abs, mut max_abs_id) = (0.0f64, 0usize);
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        let d = (f64::from(x) - f64::from(y)).abs();
        if d > max_abs {
            max_abs = d;
            max_abs_id = i;
        }
    }
    Ok(Row {
        bits_equal: a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
        max_abs,
        max_abs_id,
        kl_ab: neumaier(p.iter().zip(&q).map(|(&lp, &lq)| lp.exp() * (lp - lq))),
        kl_ba: neumaier(q.iter().zip(&p).map(|(&lq, &lp)| lq.exp() * (lq - lp))),
        tv: 0.5
            * neumaier(
                p.iter()
                    .zip(&q)
                    .map(|(&lp, &lq)| (lp.exp() - lq.exp()).abs()),
            ),
        nll_a: -p[target],
        nll_b: -q[target],
        top1_a: first_argmax(a),
        top1_b: first_argmax(b),
    })
}

#[derive(Default, Debug)]
struct Totals {
    rows: usize,
    bits_equal: usize,
    greedy_identity: usize,
    sampled_identity: usize,
    kl_ab_sum: f64,
    kl_ab_max: f64,
    kl_ba_sum: f64,
    kl_ba_max: f64,
    tv_sum: f64,
    tv_max: f64,
    max_abs: f64,
    nll_a_sum: f64,
    nll_b_sum: f64,
}

impl Totals {
    fn add(&mut self, r: &Row, sampled_same: bool) {
        self.rows += 1;
        self.bits_equal += usize::from(r.bits_equal);
        self.greedy_identity += usize::from(r.top1_a == r.top1_b);
        self.sampled_identity += usize::from(sampled_same);
        self.kl_ab_sum += r.kl_ab;
        self.kl_ab_max = self.kl_ab_max.max(r.kl_ab);
        self.kl_ba_sum += r.kl_ba;
        self.kl_ba_max = self.kl_ba_max.max(r.kl_ba);
        self.tv_sum += r.tv;
        self.tv_max = self.tv_max.max(r.tv);
        self.max_abs = self.max_abs.max(r.max_abs);
        self.nll_a_sum += r.nll_a;
        self.nll_b_sum += r.nll_b;
    }
    fn line(&self, scope: &str) -> String {
        let n = self.rows as f64;
        format!(
            concat!(
                "SUMMARY scope={} rows={} bit_equal_rows={} top1_changes={} greedy_identity={} ",
                "sampled_identity={} kl_ab_mean={:.17e} kl_ab_max={:.17e} kl_ba_mean={:.17e} ",
                "kl_ba_max={:.17e} tv_mean={:.17e} tv_max={:.17e} max_abs_logit_delta={:.17e} ",
                "nll_a_mean={:.17e} nll_b_mean={:.17e} nll_delta_mean={:.17e}"
            ),
            scope,
            self.rows,
            self.bits_equal,
            self.rows - self.greedy_identity,
            self.greedy_identity,
            self.sampled_identity,
            self.kl_ab_sum / n,
            self.kl_ab_max,
            self.kl_ba_sum / n,
            self.kl_ba_max,
            self.tv_sum / n,
            self.tv_max,
            self.max_abs,
            self.nll_a_sum / n,
            self.nll_b_sum / n,
            (self.nll_b_sum - self.nll_a_sum) / n
        )
    }
}

struct Panel {
    name: String,
    domain: String,
    template: String,
    prefix: usize,
    text: String,
}

/// Panel tape format, chosen so that Hebrew, tool-call JSON and engine source all
/// survive the round trip without an escaping layer between the pinned bytes and
/// the tokenizer: a marker line, then the panel's literal text.
///   ===PANEL name=<name> domain=<domain> template=raw|chat prefix=<n>===
fn parse_panels(text: &str) -> Vec<Panel> {
    let mut panels: Vec<Panel> = Vec::new();
    for line in text.lines() {
        if let Some(head) = line
            .strip_prefix("===PANEL ")
            .and_then(|l| l.strip_suffix("==="))
        {
            let field = |key: &str| -> String {
                head.split_whitespace()
                    .find_map(|kv| kv.strip_prefix(&format!("{key}=")))
                    .unwrap_or_else(|| panic!("panel header missing {key}: {head}"))
                    .to_owned()
            };
            panels.push(Panel {
                name: field("name"),
                domain: field("domain"),
                template: field("template"),
                prefix: field("prefix").parse().expect("panel prefix"),
                text: String::new(),
            });
        } else {
            let panel = panels
                .last_mut()
                .expect("panel text before the first ===PANEL header");
            panel.text.push_str(line);
            panel.text.push('\n');
        }
    }
    panels
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

fn row_sha(row: &[f32]) -> String {
    let mut h = Sha256::new();
    for x in row {
        h.update(x.to_bits().to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

/// One arm over one panel: prime the conditional prefix, then score `SCORED_ROWS`
/// teacher-forced steps from the panel's own tokens.
fn capture(gpu: &Dsv4Gpu, tokens: &[u32], prefix: usize) -> Vec<Vec<f32>> {
    assert_eq!(tokens.len(), prefix + SCORED_ROWS);
    let mut state = gpu
        .alloc_decode_state_for_transient(tokens.len() + PRIME_CHUNK, PRIME_CHUNK)
        .expect("fresh decode state");
    let mut draft = gpu.dspark_alloc_state().expect("draft state");
    let mut row = gpu
        .dspark_prefill_prime_chunked(&tokens[..prefix], &mut state, &mut draft, PRIME_CHUNK)
        .expect("prime");
    drop(draft);
    let mut bank = Vec::with_capacity(SCORED_ROWS);
    for step in 0..SCORED_ROWS {
        assert!(row.iter().all(|x| x.is_finite()), "finite logits");
        bank.push(row);
        if step + 1 == SCORED_ROWS {
            break;
        }
        row = gpu
            .decode_step(tokens[prefix + step], &mut state)
            .expect("teacher-forced step");
    }
    bank
}

fn bitwise_same(a: &[Vec<f32>], b: &[Vec<f32>]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits())
        })
}

/// Runs before the GPU is touched, every invocation. A comparison that cannot see a
/// planted one-bit difference cannot certify the absence of one, so this is the arm
/// that makes every PASS below non-vacuous.
fn non_vacuity_red_arm() {
    let base: Vec<f32> = (0..64).map(|i| (i as f32) * 0.03 - 1.0).collect();
    let same = compare(&base, &base, 3).expect("identical rows compare");
    assert!(
        same.bits_equal && same.max_abs == 0.0,
        "identical rows drift"
    );
    assert_eq!(same.kl_ab, 0.0);
    assert_eq!(same.kl_ba, 0.0);
    assert_eq!(same.top1_a, same.top1_b);

    let mut flipped = base.clone();
    let victim = 7;
    flipped[victim] = f32::from_bits(base[victim].to_bits() ^ 1);
    let moved = compare(&base, &flipped, 3).expect("perturbed rows compare");
    assert!(!moved.bits_equal, "one flipped bit must break bit equality");
    assert!(moved.max_abs > 0.0, "one flipped bit must move max_abs");
    assert_eq!(moved.max_abs_id, victim, "max_abs must name the flipped id");

    // A near-tie argmax flip: the #534 failure shape. It must move top-1 while
    // leaving KL small, which is precisely why a top-1 COUNT is reported next to
    // an aggregate KL rather than instead of it.
    let a = [1.0f32, 1.000_000_1, 0.0];
    let b = [1.000_000_1f32, 1.0, 0.0];
    let tie = compare(&a, &b, 2).expect("near-tie compare");
    assert_ne!(tie.top1_a, tie.top1_b, "planted near-tie flip missed");
    assert!(
        tie.kl_ab < 1e-6 && tie.kl_ba < 1e-6,
        "planted flip is small in KL"
    );
    println!(
        "RED_ARM non_vacuity=pass identical_reports_zero=true one_bit_flip_detected=true near_tie_top1_flip_detected=true small_kl_hides_it={:.17e}",
        tie.kl_ab
    );
}

/// Engagement snapshot: split-K calls plus device crossings since process start.
/// Read before and after every capture; the deltas are the proof of which arm ran.
fn dense_counts() -> (u64, u64, u64, u64) {
    let c = dense_cutlass_counts_for_gate().expect("dense cutlass counts");
    (c.splitk, c.declined, c.shapes_built, c.ws_device_flips)
}

fn delta4(before: (u64, u64, u64, u64), after: (u64, u64, u64, u64)) -> (u64, u64, u64, u64) {
    (
        after.0 - before.0,
        after.1 - before.1,
        after.2 - before.2,
        after.3 - before.3,
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(
        args.len(),
        5,
        "usage: dsv4_dense_class_gate <model-dir> <panels.txt> <panels-sha256> <new-out-dir>"
    );
    // The served program shape, asserted rather than assumed: a class answer taken
    // under a different program does not describe what a customer request runs.
    // This is the shape the engine A/B that priced this path ran under
    // (darklanes #708): PP-2, matrix experts, resident DSpark drafter.
    for (name, required) in [
        ("MEMRA_DSV4_DECODE_PATH", "device"),
        ("MEMRA_DSV4_EXPERT_ARM", "native"),
        ("MEMRA_DSV4_DENSE_ARM", "fp8"),
        ("MEMRA_DSV4_DRAFTER", "dspark"),
        ("MEMRA_DSV4_PREFILL_MOE", "reference"),
        ("MEMRA_DSV4_EP", "off"),
        ("MEMRA_MOE_F16G", "2"),
        ("MEMRA_F16G_SK", "32"),
        ("MEMRA_DSV4_VERIFY_TOPK", "device"),
        ("MEMRA_DSV4_DOTS_ARM", "f32x"),
    ] {
        assert_eq!(
            std::env::var(name).as_deref(),
            Ok(required),
            "requires {name}={required}"
        );
    }

    non_vacuity_red_arm();

    // The dense path must be PRESENT (linked archive) before any arm runs: a null
    // weak symbol reports `None`, which is a different fact from "off", and a cell
    // whose two arms are one program must fail, not report zero drift.
    match dense_cutlass_armed_for_gate() {
        None => {
            println!("CLASS_FAIL this binary does not contain the dense CUTLASS path");
            std::process::exit(1);
        }
        Some(armed) => println!("PRESENT armed_at_start={armed}"),
    }

    let dir = Path::new(&args[1]);
    let panel_bytes = std::fs::read(&args[2]).expect("panel tape");
    let panel_sha = format!("{:x}", Sha256::digest(&panel_bytes));
    assert_eq!(panel_sha, args[3], "pinned panel tape sha256");
    let panels = parse_panels(std::str::from_utf8(&panel_bytes).expect("utf-8 panel tape"));
    assert!(!panels.is_empty(), "empty panel tape");
    let output = Path::new(&args[4]);
    std::fs::create_dir(output).expect("create a new, nonexisting receipt directory");

    let tokenizer = Tokenizer::from_hf_dir(dir).expect("tokenizer");
    println!(
        "PANELS sha256={panel_sha} count={} scored_rows_per_panel={SCORED_ROWS} seed={SEED} axis=dense_cutlass armA=scalar armB=cutlass",
        panels.len(),
    );

    let mut tapes: Vec<(usize, Vec<u32>)> = Vec::new();
    for (i, panel) in panels.iter().enumerate() {
        let ids = match panel.template.as_str() {
            "raw" => tokenizer.encode(&panel.text, true),
            "chat" => tokenizer.encode_special(
                &tokenizer.apply_chat_template(&[("user", panel.text.as_str())], true),
                true,
                true,
            ),
            other => panic!("panel {i} unknown template {other}"),
        };
        assert!(
            ids.len() >= panel.prefix + SCORED_ROWS,
            "panel {} is shorter than prefix {} + {SCORED_ROWS} scored rows; no padding \
             and no repeated tokens are permitted",
            panel.name,
            panel.prefix
        );
        let tape = ids[..panel.prefix + SCORED_ROWS].to_vec();
        let mut h = Sha256::new();
        for t in &tape {
            h.update(t.to_le_bytes());
        }
        println!(
            "PANEL id={i} name={} domain={} template={} prefix={} tokens={} token_sha256={:x}",
            panel.name,
            panel.domain,
            panel.template,
            panel.prefix,
            tape.len(),
            h.finalize()
        );
        tapes.push((panel.prefix, tape));
    }

    let capacity = tapes.iter().map(|(p, _)| *p).max().unwrap() + SCORED_ROWS + PRIME_CHUNK;
    // The cell starts disarmed (arm A, scalar). Arm B is selected per panel through
    // the process-local gate setter, never by re-reading the environment.
    arm_dense_cutlass_for_gate(false).expect("stand the dense path down");
    let gpu = Dsv4Gpu::load(dir, &[0, 1], ActQuantVariant::RefFp8Round, capacity).expect("load");

    let cfg = Dsv4SampleCfg {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        seed: SEED,
    };
    let mut rows_file = create_new(&output.join("rows.jsonl"));
    let mut overall = Totals::default();

    for (i, panel) in panels.iter().enumerate() {
        let (prefix, tape) = &tapes[i];
        // A,B,B,A from one load, so the comparison is paired and nothing can drift
        // between arms. Engagement is read off the path's own counters around every
        // capture: (splitk, declined, shapes_built, flips).
        arm_dense_cutlass_for_gate(false).expect("arm A");
        let a0 = dense_counts();
        let a1 = capture(&gpu, tape, *prefix);
        let a0d = delta4(a0, dense_counts());
        assert_eq!(
            a0d.0, 0,
            "panel {}: the scalar arm ran {} split-K calls, so the arm does nothing",
            panel.name, a0d.0
        );
        arm_dense_cutlass_for_gate(true).expect("arm B");
        let b0 = dense_counts();
        let b1 = capture(&gpu, tape, *prefix);
        let b0d = delta4(b0, dense_counts());
        assert!(
            b0d.0 > 0,
            "panel {}: the cutlass arm ran no split-K calls, so the arms are one program",
            panel.name
        );
        assert!(
            b0d.3 > 0,
            "panel {}: the cutlass arm crossed no device boundary (flips=0), so the run \
             never tested the device-keyed workspace",
            panel.name
        );
        let b2 = capture(&gpu, tape, *prefix);
        arm_dense_cutlass_for_gate(false).expect("arm A repeat");
        let a2 = capture(&gpu, tape, *prefix);
        assert!(
            bitwise_same(&a1, &a2),
            "panel {}: arm A is not bit-identical to itself across repeats; \
             no class determination is possible",
            panel.name
        );
        assert!(
            bitwise_same(&b1, &b2),
            "panel {}: arm B is not bit-identical to itself across repeats; \
             no class determination is possible",
            panel.name
        );
        println!(
            "IDENTITY panel={} armA_repeat_bit_identical=true armB_repeat_bit_identical=true rows={SCORED_ROWS} axis=dense_cutlass",
            panel.name
        );
        println!(
            "ENGAGEMENT panel={} armA_splitk=0 armB_splitk={} armB_flips={} shapes_built={}",
            panel.name, b0d.0, b0d.3, b0d.2
        );

        let mut totals = Totals::default();
        for (step, (a, b)) in a1.iter().zip(&b1).enumerate() {
            let target = tape[prefix + step] as usize;
            let m = compare(a, b, target).expect("valid paired distributions");
            let pos = prefix + step;
            // The RNG stream is position-keyed and pure in (seed, pos). Banked under
            // both arms and refused on the first difference: a path that shifted
            // the stream would be a product break, not drift.
            let u = dsv4_pos_uniform(SEED, pos);
            assert_eq!(
                u.to_bits(),
                dsv4_pos_uniform(SEED, pos).to_bits(),
                "position-keyed uniform is not a pure function of (seed, pos)"
            );
            let sample_a = dsv4_sample_row(a, pos, &cfg).expect("arm A draw");
            let sample_b = dsv4_sample_row(b, pos, &cfg).expect("arm B draw");
            totals.add(&m, sample_a == sample_b);
            overall.add(&m, sample_a == sample_b);
            let line = format!(
                concat!(
                    "{{\"panel\":\"{}\",\"domain\":\"{}\",\"axis\":\"dense_cutlass\",\"prefix\":{},\"step\":{},\"position\":{},",
                    "\"vocab\":{},\"target\":{},\"bits_equal\":{},\"max_abs_logit_delta\":{:.17e},",
                    "\"max_abs_id\":{},\"kl_scalar_cutlass_nats\":{:.17e},",
                    "\"kl_cutlass_scalar_nats\":{:.17e},\"total_variation\":{:.17e},",
                    "\"scalar_nll\":{:.17e},\"cutlass_nll\":{:.17e},\"scalar_top1\":{},",
                    "\"cutlass_top1\":{},\"greedy_identical\":{},\"stream_uniform\":{:.17e},",
                    "\"scalar_sample\":{},\"cutlass_sample\":{},\"scalar_sha256\":\"{}\",",
                    "\"cutlass_sha256\":\"{}\",\"quality_verdict\":\"not_assessed\"}}"
                ),
                panel.name,
                panel.domain,
                prefix,
                step,
                pos,
                a.len(),
                target,
                m.bits_equal,
                m.max_abs,
                m.max_abs_id,
                m.kl_ab,
                m.kl_ba,
                m.tv,
                m.nll_a,
                m.nll_b,
                m.top1_a,
                m.top1_b,
                m.top1_a == m.top1_b,
                u,
                sample_a,
                sample_b,
                row_sha(a),
                row_sha(b)
            );
            writeln!(rows_file, "{line}").unwrap();
            println!("ROW {line}");
        }
        rows_file.flush().unwrap();
        println!("{}", totals.line(&format!("panel:{}", panel.name)));
    }
    println!(
        "RNG_STREAM keyed_by=(seed,position) draws_per_row=1 program_dependent=false rows={}",
        overall.rows
    );
    println!("{}", overall.line("all"));
    // NO all-equal refusal on this axis, and the difference from the expert-program
    // gate is deliberate. There, total bit-equality meant the program never engaged,
    // because engagement was read off the same grouped-route counters that the arms
    // were disjoint on. Here engagement is proven per capture by split-K calls AND
    // device crossings, which no bit pattern can fake: an all-equal result on top of
    // proven engagement would contradict the construction claim (mma fragment
    // registers per 128-k block cannot reproduce the 128-leaf smem halving tree bit
    // for bit) and would be a finding, not a vacuous pass. The expected
    // determination is new_class; same_class_candidate would demand an explanation.
    let class = if overall.bits_equal == overall.rows {
        "same_class_candidate"
    } else {
        "new_class"
    };
    println!(
        "CLASS determination={class} axis=dense_cutlass bit_equal_rows={}/{} \
         basis=paired_teacher_forced_rows_after_within_arm_bit_identity",
        overall.bits_equal, overall.rows
    );
    println!(
        "COMPLETE class determination and drift rows; no quality admission, no threshold, \
         no serving decision, no performance claim"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_and_shifted_rows_carry_zero_divergence_both_directions() {
        let a = [1.0f32, 0.0, -2.0];
        for b in [a, [1001.0, 1000.0, 998.0]] {
            let m = compare(&a, &b, 1).unwrap();
            assert_eq!(m.kl_ab, 0.0);
            assert_eq!(m.kl_ba, 0.0);
            assert_eq!(m.tv, 0.0);
            assert_eq!(m.nll_a, m.nll_b);
        }
    }

    #[test]
    fn asymmetric_kl_is_reported_in_both_directions() {
        let m = compare(&[0.0, 0.0], &[3.0f32.ln(), 0.0], 0).unwrap();
        assert!((m.kl_ab - 0.5 * (4.0f64 / 3.0).ln()).abs() < 1e-7);
        assert!((m.kl_ba - (0.75 * 1.5f64.ln() + 0.25 * 0.5f64.ln())).abs() < 1e-7);
        assert!(
            m.kl_ab != m.kl_ba,
            "a symmetric report would hide direction"
        );
        assert!((m.tv - 0.25).abs() < 1e-7);
    }

    #[test]
    fn bit_equality_is_stricter_than_distribution_equality() {
        // A shift leaves every probability alone, so KL and TV are zero while the
        // raw logits differ in every bit. Reporting only KL would call this
        // same-class; the class question is about the tree, so bits decide.
        let a = [1.0f32, 0.0];
        let b = [1001.0f32, 1000.0];
        let m = compare(&a, &b, 0).unwrap();
        assert_eq!(m.kl_ab, 0.0);
        assert!(!m.bits_equal);
        assert!(m.max_abs > 0.0);
    }

    #[test]
    fn argmax_takes_the_first_maximum_on_a_tie() {
        assert_eq!(first_argmax(&[-0.0, 0.0, -1.0]), 0);
        assert_eq!(first_argmax(&[1.0, 1.0]), 0);
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
            assert!(compare(a, b, target).is_err());
        }
    }

    #[test]
    fn the_non_vacuity_red_arm_passes_on_this_comparison() {
        non_vacuity_red_arm();
    }

    #[test]
    fn the_panel_tape_round_trips_name_domain_template_and_prefix() {
        let panels = parse_panels(
            "===PANEL name=a domain=code template=raw prefix=8===\nfn main() {}\n\
             ===PANEL name=b domain=chat template=chat prefix=16===\nשלום\n",
        );
        assert_eq!(panels.len(), 2);
        assert_eq!(panels[0].prefix, 8);
        assert_eq!(panels[0].text, "fn main() {}\n");
        assert_eq!(panels[1].domain, "chat");
        assert_eq!(panels[1].template, "chat");
        assert_eq!(panels[1].text, "שלום\n");
    }

    #[test]
    fn totals_report_max_alongside_mean_in_both_directions() {
        let mut t = Totals::default();
        for (kl_ab, kl_ba) in [(1.0, 3.0), (0.0, 0.0)] {
            t.add(
                &Row {
                    bits_equal: false,
                    max_abs: kl_ab,
                    max_abs_id: 0,
                    kl_ab,
                    kl_ba,
                    tv: 0.0,
                    nll_a: 0.0,
                    nll_b: 0.0,
                    top1_a: 0,
                    top1_b: 1,
                },
                false,
            );
        }
        let line = t.line("t");
        assert!(line.contains("top1_changes=2"));
        assert!(line.contains("kl_ab_max=1.00000000000000000e0"));
        assert!(line.contains("kl_ba_max=3.00000000000000000e0"));
    }
}
