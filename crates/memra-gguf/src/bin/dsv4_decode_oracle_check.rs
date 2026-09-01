//! dsv4-decode-oracle-check: lane-6 gates (a2) + (b) SEMANTIC verdicts against the CPU
//! oracle (the lane-4 teacher-forcing instrument, factor 1). Policy banked in
//! wt-dsv4-loader research/dsv4-flash-loader-20260818/RECEIPTS.md "Lane 6" run-1
//! corrections BEFORE this binary existed.
//!
//!   (a2) at every decode checkpoint, the GPU decode logits row must sit within the
//!        lane-4 derived bf16 bound of the CPU oracle row computed teacher-forced on
//!        the decode trajectory: thr4 = u·√86·√(2·ln n)·absmax_cpu (u = 2⁻⁸); argmax
//!        must agree except in-band near-ties (band4 = 3·√2·u·√86·|top1|); top-5 and
//!        top-20 set changes confined to band4 of the respective boundary.
//!   (b)  corrected greedy verdict: (i) decode's FIRST divergence from the lane-4
//!        banked sequence must be an in-band near-tie on the CPU row (both candidates
//!        within band4 of each other at the top); (ii) over the FULL decode
//!        trajectory, every CPU-argmax disagreement must be in-band (raw counts and
//!        margins always printed, never hidden).
//!
//! Inputs: the decode gate's out-dir (decode_gate.json + decode_ckpt_logits.bin), the
//! CPU verifier's out-dir (cpu_logits_all.bin), the lane-4 banked greedy json.
//!
//! Usage: dsv4-decode-oracle-check <decode-out-dir> <cpu-verify-dir> <lane4-greedy.json>
//!        exit 0 = (a2) and (b) PASS

use memra_gguf::config::JsonObj;
use memra_gguf::dsv4_forward::{drift_coeff, expert_arm_native};
use std::path::Path;

/// Lane 7: the GPU-vs-CPU-oracle bound of the ACTIVE class as a SUM of each
/// realization's own coefficient (triangle through the f32 ideal). bf16 arm: the CPU
/// f64 oracle carries zero noise → u_b·√86, factor 1 (lane-6 unchanged). Native arm:
/// GPU √(86u_b²+86u_q²) + CPU-quantized-oracle √86·u_q (both sides realize the
/// quantizers independently — RECEIPTS.md "Lane 7").
fn pair_coeff() -> f64 {
    if expert_arm_native() {
        drift_coeff(86.0, 86.0) + drift_coeff(0.0, 86.0)
    } else {
        drift_coeff(86.0, 0.0)
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}

fn top_ids(v: &[f32], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..v.len()).collect();
    order.sort_by(|&a, &b| {
        v[b].partial_cmp(&v[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    order.truncate(k);
    order
}

fn read_f32_bin(path: &Path) -> Vec<f32> {
    let raw = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!(
            "usage: dsv4-decode-oracle-check <decode-out-dir> <cpu-verify-dir> <lane4-greedy.json>"
        );
        std::process::exit(2);
    }
    let dec_dir = Path::new(&args[1]);
    let cpu_dir = Path::new(&args[2]);
    let lane4 = JsonObj::parse(&std::fs::read_to_string(&args[3]).expect("lane4 json"));
    let lane4_tokens = lane4.u32_array("tokens_run0").expect("lane4 tokens_run0");

    let dg = JsonObj::parse(
        &std::fs::read_to_string(dec_dir.join("decode_gate.json")).expect("decode_gate.json"),
    );
    let prompt = dg.u32_array("prompt").expect("prompt");
    let tokens = dg.u32_array("tokens_run0").expect("tokens_run0");
    let ckpts: Vec<usize> = dg
        .u32_array("checkpoints")
        .expect("checkpoints")
        .into_iter()
        .map(|x| x as usize)
        .collect();
    let n_new = tokens.len();

    let cpu = read_f32_bin(&cpu_dir.join("cpu_logits_all.bin"));
    assert_eq!(cpu.len() % n_new, 0, "cpu rows vs n_new");
    let vocab = cpu.len() / n_new;
    let dec = read_f32_bin(&dec_dir.join("decode_ckpt_logits.bin"));
    assert_eq!(dec.len(), ckpts.len() * vocab, "decode ckpt rows");
    let ev = (2.0f64 * (vocab as f64).ln()).sqrt();
    let band4 = |x: f64| 3.0 * 2f64.sqrt() * pair_coeff() * x.abs();
    println!(
        "class coefficient C = {:.4e} [{}]",
        pair_coeff(),
        if expert_arm_native() {
            "NATIVE expert arm (pair: GPU + quantized oracle)"
        } else {
            "bf16-dequant arm (factor 1 vs the f64 oracle)"
        }
    );
    println!(
        "dsv4-decode-oracle-check | prompt {} | decode tokens {n_new} | vocab {vocab} | checkpoints {}",
        prompt.len(),
        ckpts.len()
    );

    // ---- (a2): decode rows vs CPU oracle rows at the checkpoints
    // decode row at length s predicts position s == cpu row index s - prompt_len
    let mut a2 = true;
    let mut worst = (0f64, 0usize);
    println!("GATE (a2) decode vs CPU ORACLE (thr4 = u·√86·√(2 ln n)·absmax_cpu, factor 1):");
    println!(
        "| s | max-abs | thr4 | argmax d/c | in-band | top5 ovl/viol | top20 ovl/viol | verdict |"
    );
    let n_rows = cpu.len() / vocab;
    for (ci, &s) in ckpts.iter().enumerate() {
        let drow = &dec[ci * vocab..(ci + 1) * vocab];
        let ridx = s - prompt.len();
        if ridx >= n_rows {
            // the final checkpoint predicts the token AFTER the verified sequence —
            // no CPU row exists for it (teacher-forcing stops at s-2). Not a miss:
            // reported, and the remaining checkpoints stay >= the mandated 40.
            println!("| {s} | - | - | - | - | - | - | NO CPU ROW (past sequence end) |");
            continue;
        }
        let crow = &cpu[ridx * vocab..(ridx + 1) * vocab];
        let mut max_abs = 0f64;
        let mut absmax = 0f32;
        for (&g, &r) in drow.iter().zip(crow) {
            max_abs = max_abs.max((g as f64 - r as f64).abs());
            absmax = absmax.max(r.abs());
        }
        let thr = pair_coeff() * ev * absmax as f64;
        let ad = argmax(drow);
        let ac = argmax(crow);
        let argmax_ok = ad == ac || {
            let margin = crow[ac] as f64 - crow[ad] as f64;
            margin <= band4(crow[ac] as f64)
        };
        let mut viol5 = 0usize;
        let mut viol20 = 0usize;
        let mut ov5 = 0usize;
        let mut ov20 = 0usize;
        for (k, viol, ov) in [(5usize, &mut viol5, &mut ov5), (20, &mut viol20, &mut ov20)] {
            let td: std::collections::BTreeSet<usize> = top_ids(drow, k).into_iter().collect();
            let tc_v = top_ids(crow, k);
            let tc: std::collections::BTreeSet<usize> = tc_v.iter().cloned().collect();
            let boundary = crow[tc_v[k - 1]] as f64;
            *ov = td.intersection(&tc).count();
            *viol = td
                .symmetric_difference(&tc)
                .filter(|&&id| (crow[id] as f64 - boundary).abs() > band4(boundary))
                .count();
        }
        let ok = max_abs <= thr && argmax_ok && viol5 == 0 && viol20 == 0;
        if max_abs > worst.0 {
            worst = (max_abs, s);
        }
        if !ok {
            a2 = false;
        }
        println!(
            "| {s} | {max_abs:.3e} | {thr:.3e} | {ad}/{ac} | {} | {ov5}/5 v{viol5} | {ov20}/20 v{viol20} | {} |",
            if ad == ac {
                "same".into()
            } else {
                format!(
                    "flip({}, margin {:.4})",
                    if argmax_ok { "in-band" } else { "OUT" },
                    crow[ac] - crow[ad]
                )
            },
            if ok { "PASS" } else { "FAIL" }
        );
    }
    println!(
        "(a2) worst max-abs {:.3e} at s={} (lane-4's own (a) measured 4.35 vs 6.71 on this bound)",
        worst.0, worst.1
    );

    // ---- (b) corrected greedy verdict
    // (i) first divergence vs the lane-4 banked sequence: in-band near-tie on the CPU row
    let n_cmp = lane4_tokens.len().min(tokens.len()).min(160);
    let first_div = (0..n_cmp).find(|&i| tokens[i] != lane4_tokens[i]);
    let mut b_ok = true;
    match first_div {
        None => println!(
            "GATE (b) part (i): decode == lane-4 banked for all {n_cmp} compared steps (identity holds)"
        ),
        Some(i) => {
            // cpu row i predicts step i's token (teacher-forced on the DECODE trajectory,
            // whose prefix equals lane-4's up to the first divergence)
            let crow = &cpu[i * vocab..(i + 1) * vocab];
            let c_dec = tokens[i] as usize;
            let c_l4 = lane4_tokens[i] as usize;
            let top1 = argmax(crow);
            let margin = (crow[c_l4] as f64 - crow[c_dec] as f64).abs();
            let band = band4(crow[top1] as f64);
            let ok = margin <= band;
            if !ok {
                b_ok = false;
            }
            println!(
                "GATE (b) part (i): first divergence at step {i}: decode {c_dec} vs lane4 {c_l4}; CPU logits {:.4} vs {:.4} (margin {:.4}, band {:.4}, cpu top1 {top1}) -> {}",
                crow[c_dec],
                crow[c_l4],
                margin,
                band,
                if ok {
                    "in-band near-tie (legitimate realization flip)"
                } else {
                    "OUT OF BAND — REAL BUG"
                }
            );
        }
    }
    // (ii) full-trajectory CPU-argmax agreement with in-band-only disagreements
    let mut agree = 0usize;
    let mut flips: Vec<(usize, f64, f64)> = Vec::new();
    for i in 0..n_new {
        let crow = &cpu[i * vocab..(i + 1) * vocab];
        let want = tokens[i] as usize;
        let top1 = argmax(crow);
        if top1 == want {
            agree += 1;
        } else {
            let margin = crow[top1] as f64 - crow[want] as f64;
            let band = band4(crow[top1] as f64);
            if margin > band {
                b_ok = false;
            }
            flips.push((i, margin, band));
        }
    }
    println!(
        "GATE (b) part (ii): CPU teacher-forcing over the decode trajectory: {agree}/{n_new} agree; {} disagreements:",
        flips.len()
    );
    for (i, margin, band) in &flips {
        println!(
            "  step {i}: cpu margin {margin:.4} vs band {band:.4} -> {}",
            if margin <= band {
                "in-band"
            } else {
                "OUT OF BAND — REAL BUG"
            }
        );
    }

    println!(
        "DSV4 DECODE ORACLE CHECK: {} (a2 {a2} | b {b_ok})",
        if a2 && b_ok { "PASS" } else { "FAIL" }
    );
    std::process::exit(if a2 && b_ok { 0 } else { 1 });
}
