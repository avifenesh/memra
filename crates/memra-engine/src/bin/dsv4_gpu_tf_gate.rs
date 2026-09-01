//! dsv4-gpu-tf-gate: GPU teacher-forced token-id agreement vs a BANKED CPU-oracle greedy
//! trajectory (0731 re-gate, publish-gate item 2 instrument).
//!
//! The realization-stability doctrine (lane-6, banked): the CPU oracle / banked
//! trajectory is the semantic truth instrument; raw greedy identity between two GPU
//! realizations is a coin-flip at in-band near-ties. This gate therefore TEACHER-FORCES
//! the banked token sequence through the GPU decode path (always feeding the banked
//! token, never the GPU pick) and compares the GPU argmax at every position against the
//! banked next token — 160 independent per-position checks, no divergence cascade.
//!
//! In-band classification (the dsv4-greedy-verify native-class rule, verbatim): a
//! disagreement is a legitimate realization flip iff the CPU margin between the CPU
//! top-1 (== the banked token) and the GPU's pick is within
//!   band = 3·√2·(C_gpu + C_cpu)·|cpu top1|,   C = drift_coeff per the lane-7 doctrine.
//! The banked trajectory carries per-step CPU top-8 ids+logits (flat aux JSON generated
//! from the Gate C bank; source sha printed); a GPU pick outside the CPU top-8 can only
//! be bounded from below (top1 − top8 logit) — if that lower bound exceeds the band it
//! is OUT-OF-BAND definitively; otherwise the position is UNRESOLVED-BY-THE-BANK and
//! the gate FAILS loudly (CPU adjudication owed; the GPU row is banked). A skip or an
//! unresolved row is never a PASS.
//!
//! Determinism: the full teacher-forced pass runs TWICE (fresh decode state each);
//! token verdicts and the 160-row logits stream sha256 must be identical.
//!
//! Usage: dsv4-gpu-tf-gate <model-dir> <tf.json> <out-dir> [dev0,dev1]
//!   exit 0 = every position agrees or is an in-band near-tie, determinism holds

use memra_engine::dsv4_gpu::Dsv4Gpu;
use memra_gguf::config::JsonObj;
use memra_gguf::dsv4_forward::{ActQuantVariant, drift_coeff, expert_arm_native};
use std::io::Write as _;
use std::path::Path;

/// dsv4-greedy-verify's native-class band (verbatim rule): GPU native realization vs the
/// CPU oracle — pair coefficient C_gpu + C_cpu at head depth 86.
fn native_band(top1: f32) -> f64 {
    let c_pair = drift_coeff(86.0, 86.0) + drift_coeff(0.0, 86.0);
    3.0 * 2f64.sqrt() * c_pair * (top1.abs() as f64)
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best as u32
}

fn vram_line(gpu: &Dsv4Gpu, tag: &str) {
    if let Ok(rows) = gpu.vram_report() {
        let line = rows
            .iter()
            .map(|(dev, free, total, _)| {
                format!(
                    "dev{dev} used {:.2}/{:.2} GiB (free {:.2})",
                    (*total - *free) as f64 / 2f64.powi(30),
                    *total as f64 / 2f64.powi(30),
                    *free as f64 / 2f64.powi(30)
                )
            })
            .collect::<Vec<_>>()
            .join(" | ");
        println!("[vram {tag}] {line}");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: dsv4-gpu-tf-gate <model-dir> <tf.json> <out-dir> [dev0,dev1]");
        std::process::exit(2);
    }
    let t0 = std::time::Instant::now();
    let dir = Path::new(&args[1]);
    let tf = JsonObj::parse(&std::fs::read_to_string(&args[2]).expect("read tf json"));
    let out_dir = Path::new(&args[3]);
    std::fs::create_dir_all(out_dir).expect("mkdir out");
    let devices: Vec<usize> = args
        .get(4)
        .map(|s| s.split(',').map(|x| x.parse().expect("device")).collect())
        .unwrap_or_else(|| vec![0, 1]);

    let variant_tag = tf.string("variant").expect("tf json: variant");
    let variant = ActQuantVariant::from_fixture_tag(&variant_tag);
    let prompt = tf.u32_array("prompt").expect("tf json: prompt");
    let tokens = tf.u32_array("tokens").expect("tf json: tokens");
    let top8_ids = tf.u32_array("top8_ids").expect("tf json: top8_ids");
    let top8_logits = tf.f32_array("top8_logits").expect("tf json: top8_logits");
    let n = tokens.len();
    assert_eq!(top8_ids.len(), n * 8, "top8_ids must be n*8 flat");
    assert_eq!(top8_logits.len(), n * 8, "top8_logits must be n*8 flat");
    // bank integrity: the CPU top-1 at every step IS the banked token (greedy), and the
    // top-8 logits are non-increasing.
    for i in 0..n {
        assert_eq!(
            top8_ids[i * 8],
            tokens[i],
            "bank corrupt: step {i} cpu top1 != banked token"
        );
        for k in 1..8 {
            assert!(
                top8_logits[i * 8 + k] <= top8_logits[i * 8 + k - 1],
                "bank corrupt: step {i} top8 logits not sorted"
            );
        }
    }
    println!(
        "dsv4-gpu-tf-gate | model {} | variant {variant_tag} | banked source sha {} | prompt {} + {} banked tokens | devices {devices:?}",
        dir.display(),
        tf.string("source_sha256").unwrap_or_default(),
        prompt.len(),
        n
    );
    println!(
        "seams: MEMRA_DSV4_EXPERT_ARM='{}' MEMRA_DSV4_DECODE_PATH='{}' MEMRA_DSV4_DOTS_ARM='{}'",
        std::env::var("MEMRA_DSV4_EXPERT_ARM").unwrap_or_default(),
        std::env::var("MEMRA_DSV4_DECODE_PATH").unwrap_or_default(),
        std::env::var("MEMRA_DSV4_DOTS_ARM").unwrap_or_default(),
    );
    if !expert_arm_native() {
        println!("note: bf16-dequant expert arm active (native is the shipping stack)");
    }

    let max_seq = prompt.len() + n + 96;
    let gpu = Dsv4Gpu::load(dir, &devices, variant, max_seq).expect("load");
    println!(
        "loaded: split at layer {}, t={:.0}s",
        gpu.split_at,
        t0.elapsed().as_secs_f64()
    );
    vram_line(&gpu, "post-load");

    let band_hdr = native_band(1.0);
    println!(
        "band rule: 3·√2·(C_gpu + C_cpu)·|cpu top1| = {band_hdr:.4}·|top1| (dsv4-greedy-verify native class)"
    );

    let mut run_shas: Vec<String> = Vec::new();
    let mut run_verdicts: Vec<Vec<u32>> = Vec::new(); // gpu argmax per position per run
    let mut timing_ms: Vec<f64> = Vec::new();
    for run in 0..2usize {
        use sha2::{Digest, Sha256};
        let mut state = gpu.alloc_decode_state().expect("alloc decode state");
        let pre = gpu
            .prefill_with_cache(&prompt, &mut state)
            .expect("prefill_with_cache");
        if run == 0 {
            vram_line(&gpu, "post-prefill");
        }
        let mut hasher = Sha256::new();
        let mut picks: Vec<u32> = Vec::with_capacity(n);
        let mut logits = pre.logits;
        for (i, &banked_tok) in tokens.iter().enumerate() {
            for v in &logits {
                hasher.update(v.to_le_bytes());
            }
            let pick = argmax(&logits);
            picks.push(pick);
            if run == 0 && pick != banked_tok {
                // bank the GPU row at every disagreement (adjudication material)
                let mut blob = Vec::with_capacity(logits.len() * 4);
                for v in &logits {
                    blob.extend_from_slice(&v.to_le_bytes());
                }
                std::fs::write(out_dir.join(format!("gpu_logits_step{i}.bin")), &blob)
                    .expect("bank gpu row");
            }
            if i + 1 < n {
                let st0 = std::time::Instant::now();
                logits = gpu
                    .decode_step(banked_tok, &mut state)
                    .expect("decode_step");
                if run == 0 {
                    timing_ms.push(st0.elapsed().as_secs_f64() * 1000.0);
                }
            }
        }
        let sha: String = hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        println!(
            "[run {run}] {} teacher-forced positions in {:.1}s | logits stream sha256 {sha}",
            n,
            t0.elapsed().as_secs_f64()
        );
        vram_line(&gpu, &format!("post-run{run}"));
        run_shas.push(sha);
        run_verdicts.push(picks);
    }

    // determinism
    let det = run_shas[0] == run_shas[1] && run_verdicts[0] == run_verdicts[1];
    println!(
        "determinism: picks {} | logits streams {} (sha {})",
        if run_verdicts[0] == run_verdicts[1] {
            "IDENTICAL"
        } else {
            "DIVERGENT"
        },
        if run_shas[0] == run_shas[1] {
            "BYTE-IDENTICAL"
        } else {
            "DIVERGENT"
        },
        run_shas[0]
    );

    // per-position verdicts (run 0; runs proven identical above)
    let mut agree = 0usize;
    let mut in_band = 0usize;
    let mut out_of_band = 0usize;
    let mut unresolved = 0usize;
    let mut detail = String::new();
    for i in 0..n {
        let pick = run_verdicts[0][i];
        let want = tokens[i];
        if pick == want {
            agree += 1;
            continue;
        }
        let top1 = top8_logits[i * 8];
        let band = native_band(top1);
        let row_ids = &top8_ids[i * 8..i * 8 + 8];
        let row_lg = &top8_logits[i * 8..i * 8 + 8];
        let (klass, margin_str) = match row_ids.iter().position(|&id| id == pick) {
            Some(k) => {
                let margin = (top1 - row_lg[k]) as f64;
                if margin <= band {
                    in_band += 1;
                    ("in-band near-tie", format!("{margin:.4}"))
                } else {
                    out_of_band += 1;
                    ("OUT-OF-BAND — REAL BUG", format!("{margin:.4}"))
                }
            }
            None => {
                let lb = (top1 - row_lg[7]) as f64;
                if lb > band {
                    out_of_band += 1;
                    (
                        "OUT-OF-BAND (beyond cpu top-8, lower bound exceeds band)",
                        format!(">{lb:.4}"),
                    )
                } else {
                    unresolved += 1;
                    (
                        "UNRESOLVED-BY-THE-BANK (beyond cpu top-8; CPU adjudication owed — FAIL, never a pass)",
                        format!(">{lb:.4}"),
                    )
                }
            }
        };
        let line = format!(
            "  disagreement at step {i}: gpu {pick} vs banked {want} | cpu margin {margin_str} vs band {band:.4} -> {klass}"
        );
        println!("{line}");
        detail.push_str(&line);
        detail.push('\n');
    }

    let mean_ms = if timing_ms.is_empty() {
        f64::NAN
    } else {
        timing_ms.iter().sum::<f64>() / timing_ms.len() as f64
    };
    println!(
        "informational (single-run, NOT a perf claim): mean {mean_ms:.1} ms/step over {} teacher-forced decode steps",
        timing_ms.len()
    );

    let pass = det && out_of_band == 0 && unresolved == 0;
    let mut f = std::fs::File::create(out_dir.join("tf_gate.json")).expect("json");
    write!(
        f,
        "{{\n  \"variant\": \"{variant_tag}\",\n  \"positions\": {n},\n  \"agree\": {agree},\n  \"in_band\": {in_band},\n  \"out_of_band\": {out_of_band},\n  \"unresolved\": {unresolved},\n  \"determinism\": {det},\n  \"stream_sha256\": [\"{}\", \"{}\"],\n  \"pass\": {pass}\n}}\n",
        run_shas[0], run_shas[1]
    )
    .expect("write json");

    println!(
        "\nGPU TEACHER-FORCING GATE [{variant_tag}]: {} — {agree}/{n} agree, {in_band} in-band near-tie(s), {out_of_band} out-of-band, {unresolved} unresolved | determinism {} | elapsed {:.0}s",
        if pass { "PASS" } else { "FAIL" },
        det,
        t0.elapsed().as_secs_f64()
    );
    std::process::exit(if pass { 0 } else { 1 });
}
