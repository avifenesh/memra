//! dsv4-gpu-decode-gate: lane-6 decode-path gates against the lane-4 re-prefill path.
//! Design + doctrine + RUN-1 CORRECTIONS banked in wt-dsv4-loader
//! research/dsv4-flash-loader-20260818/RECEIPTS.md "Lane 6" BEFORE each run.
//!
//! Measured facts the corrected policy rests on (run-1 bisect, banked): the pure
//! lane-4 path is not realization-stable — appending ONE token moves the same logits
//! row by 0.18–3.08 max-abs (the m-floor; per-GEMM row m-sensitivity is only ≤1.2e-7
//! but the trunk amplifies to a decorrelation floor) — so decode-vs-reprefill deltas
//! are bounded by the triangle inequality through the CPU oracle, NOT by a per-hop
//! reorder walk:
//!   thr_a1(row) = 2 · thr4,  thr4 = u·√86·√(2·ln n)·absmax_ref  (u = 2⁻⁸, lane-4 (a))
//! and every discrete disagreement (argmax / top-5 / top-20) must be in the PAIR band
//! 2·band4, band4 = 3·√2·u·√86·|ref boundary logit| (lane-4 run-3 band rule). The
//! SEMANTIC instrument (factor 1 vs the CPU oracle) is the separate
//! dsv4-decode-oracle-check over the banked cpu teacher-forcing rows.
//!
//! Gates here: (a1) decode-vs-reprefill at checkpoint lengths (corrected thresholds,
//! m-floor rows printed as context), (b-raw) decode tokens vs the lane-4 banked greedy
//! (raw counts + first divergence; the corrected (b) verdict needs the CPU rows),
//! (c) long-probe coverage, (d) two-run byte determinism, (e) VRAM/cache math.
//! Timing lines are informational, single-run, NOT perf claims.
//!
//! Usage: dsv4-gpu-decode-gate <model-dir> <fixtures.json> <lane4-greedy.json> <out-dir>
//!        [n_new] [dev0,dev1]                exit 0 = gates a1/c/d/e PASS

use memra_engine::dsv4_gpu::Dsv4Gpu;
use memra_gguf::config::JsonObj;
use memra_gguf::dsv4_forward::{FixtureSpec, drift_coeff, expert_arm_native};
use std::io::Write as _;
use std::path::Path;

/// Lane 7: per-realization drift coefficient of the ACTIVE numeric class at the head
/// (d = 86). bf16 arm: u_b·√86 (lane-6 unchanged). Native arm: √(86·u_b² + 86·u_q²)
/// (RECEIPTS.md "Lane 7"). (a1) compares two GPU realizations of the same arm →
/// pair bound 2·C in both classes.
fn head_coeff() -> f64 {
    if expert_arm_native() {
        drift_coeff(86.0, 86.0)
    } else {
        drift_coeff(86.0, 0.0)
    }
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

fn top_ids(v: &[f32], k: usize) -> Vec<u32> {
    let mut order: Vec<usize> = (0..v.len()).collect();
    order.sort_by(|&a, &b| {
        v[b].partial_cmp(&v[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    order.into_iter().take(k).map(|x| x as u32).collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

fn vram_line(gpu: &Dsv4Gpu, tag: &str) {
    let line = match gpu.vram_report() {
        Ok(rows) => rows
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
            .join(" | "),
        Err(err) => format!("vram report failed: {err}"),
    };
    println!("[vram {tag}] {line}");
}

/// Band-rule check on a set difference: every id in the symmetric difference of the
/// two top-k sets must lie within `band` of the reference rank-k boundary logit.
fn band_violations(dec: &[f32], rl: &[f32], k: usize, band: f64) -> (usize, usize) {
    let tkd: std::collections::BTreeSet<u32> = top_ids(dec, k).into_iter().collect();
    let tkr_v = top_ids(rl, k);
    let tkr: std::collections::BTreeSet<u32> = tkr_v.iter().cloned().collect();
    let boundary = rl[tkr_v[k - 1] as usize] as f64;
    let overlap = tkd.intersection(&tkr).count();
    let viol = tkd
        .symmetric_difference(&tkr)
        .filter(|&&id| (rl[id as usize] as f64 - boundary).abs() > band)
        .count();
    (overlap, viol)
}

#[allow(clippy::manual_checked_ops)] // allow: the explicit zero guard names the degenerate-ratio case; checked ops would hide the sentinel
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!(
            "usage: dsv4-gpu-decode-gate <model-dir> <fixtures.json> <lane4-greedy.json> \
             <out-dir> [n_new] [dev0,dev1]"
        );
        std::process::exit(2);
    }
    let t0 = std::time::Instant::now();
    let dir = Path::new(&args[1]);
    let spec = FixtureSpec::load(Path::new(&args[2]));
    let lane4_txt = std::fs::read_to_string(&args[3]).expect("read lane4 greedy json");
    let lane4 = JsonObj::parse(&lane4_txt);
    let lane4_tokens = lane4
        .u32_array("tokens_run0")
        .expect("lane4 json: tokens_run0");
    let out_dir = Path::new(&args[4]);
    std::fs::create_dir_all(out_dir).expect("mkdir out");
    let n_new: usize = args
        .get(5)
        .map(|x| x.parse().expect("n_new"))
        .unwrap_or(2176);
    let devices: Vec<usize> = args
        .get(6)
        .map(|s| s.split(',').map(|x| x.parse().expect("device")).collect())
        .unwrap_or_else(|| vec![0, 1]);

    let prompt = spec.tokens_32.clone();
    let s_max = prompt.len() + n_new;
    let max_seq = s_max + 96;
    println!(
        "dsv4-gpu-decode-gate | model {} | variant {} | n_new {n_new} (s_max {s_max}) | max_seq {max_seq} | devices {devices:?}",
        dir.display(),
        spec.variant_tag
    );

    // checkpoint lengths (receipts protocol): 40 consecutive early + boundary straddles
    let mut checkpoints: Vec<usize> = (33..=72).collect();
    checkpoints.extend_from_slice(&[
        126, 127, 128, 129, 130, 131, 132, 160, 192, 255, 256, 257, 384, 512, 768, 1024, 1025,
        2048, 2052, 2053, 2056, 2176, 2208,
    ]);
    checkpoints.retain(|&s| s > prompt.len() && s <= s_max);
    let saturation_reached = s_max >= 2052;
    println!(
        "checkpoints: {} lengths; top-512 saturation (s >= 2052) {}",
        checkpoints.len(),
        if saturation_reached {
            "REACHED"
        } else {
            "NOT reachable at this length (say so, per the lane brief)"
        }
    );

    let gpu = Dsv4Gpu::load(dir, &devices, spec.variant, max_seq).expect("load");
    println!(
        "loaded: split at layer {}, t={:.0}s",
        gpu.split_at,
        t0.elapsed().as_secs_f64()
    );
    vram_line(&gpu, "post-load");

    // design-formula cache bytes per device (independent re-statement of the receipts
    // math; must equal the allocator's measured bytes exactly — gate (e))
    let expected_bytes: Vec<u64> = {
        let d = gpu.model.cfg();
        let mc = &gpu.model.mc;
        let win = d.sliding_window as usize;
        let hd = d.head_dim as usize;
        let ihd = d.index_head_dim as usize;
        let n_trunk = mc.n_layer - mc.nextn_predict_layers;
        let mut per_dev = vec![0u64; devices.len()];
        for il in 0..n_trunk {
            let stage = gpu.layer_stage[il as usize];
            let ratio = d.compress_ratio(il) as usize;
            let mut b = 0u64;
            if ratio == 0 {
                b += (win * hd * 4) as u64;
            } else {
                let cap = max_seq / ratio;
                b += ((win + cap) * hd * 4) as u64;
                let (coff, lat) = if ratio == 4 { (2, 2 * hd) } else { (1, hd) };
                b += (2 * coff * ratio * lat * 4) as u64;
                if d.has_indexer(il) {
                    b += (cap * ihd * 4) as u64;
                    let ilat = 2 * ihd; // fine => coff 2
                    b += (2 * 2 * ratio * ilat * 4) as u64;
                }
            }
            per_dev[stage] += b;
        }
        per_dev
    };

    let mut run_tokens: Vec<Vec<u32>> = Vec::new();
    let mut run_shas: Vec<String> = Vec::new();
    let mut ckpt_decode: Vec<(usize, Vec<f32>)> = Vec::new();
    let mut timing_200 = Vec::new();
    let mut timing_1024 = Vec::new();
    let mut cache_alloc_measured: Vec<u64> = Vec::new();

    for run in 0..2usize {
        use sha2::{Digest, Sha256};
        let mut state = gpu.alloc_decode_state().expect("alloc decode state");
        if run == 0 {
            cache_alloc_measured = state.cache_bytes.clone();
            vram_line(&gpu, "post-alloc");
            println!(
                "[cache] allocated bytes per device: {:?} | design formula: {:?} | {}",
                state.cache_bytes,
                expected_bytes,
                if state.cache_bytes == expected_bytes {
                    "MATCH"
                } else {
                    "MISMATCH"
                }
            );
        }
        let run_t0 = std::time::Instant::now();
        let mut ids = prompt.clone();
        let pre = gpu
            .prefill_with_cache(&ids, &mut state)
            .expect("prefill_with_cache");
        if run == 0 {
            vram_line(&gpu, "post-prefill");
            println!(
                "[run {run}] prefill s={} done t={:.1}s",
                ids.len(),
                run_t0.elapsed().as_secs_f64()
            );
        }
        let mut hasher = Sha256::new();
        let mut tokens: Vec<u32> = Vec::with_capacity(n_new);
        let mut tok = argmax(&pre.logits);
        for step in 0..n_new {
            let st0 = std::time::Instant::now();
            let logits = gpu.decode_step(tok, &mut state).expect("decode_step");
            let ms = st0.elapsed().as_secs_f64() * 1000.0;
            let s_now = state.pos; // sequence length after this step
            if run == 0 {
                if (190..=210).contains(&s_now) {
                    timing_200.push(ms);
                }
                if (1014..=1034).contains(&s_now) {
                    timing_1024.push(ms);
                }
            }
            tokens.push(tok);
            ids.push(tok);
            for v in &logits {
                hasher.update(v.to_le_bytes());
            }
            if run == 0 && checkpoints.contains(&s_now) {
                ckpt_decode.push((s_now, logits.clone()));
            }
            tok = argmax(&logits);
            if step % 200 == 0 || step + 1 == n_new {
                println!(
                    "[run {run}] step {step}: s={} tok {} ({:.1} ms/step) t={:.1}s",
                    s_now,
                    tokens[step],
                    ms,
                    run_t0.elapsed().as_secs_f64()
                );
            }
        }
        let sha: String = hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        println!(
            "[run {run}] {} steps in {:.1}s (informational, single-run, not a perf claim); logits stream sha256 {}",
            n_new,
            run_t0.elapsed().as_secs_f64(),
            sha
        );
        vram_line(&gpu, &format!("post-run{run}"));
        run_tokens.push(tokens);
        run_shas.push(sha);

        if run == 0 {
            // bank the sequence for the CPU teacher-forcing verifier IMMEDIATELY
            // (spot discipline + lets the CPU run start while run 1 + compares proceed)
            let mut f = std::fs::File::create(out_dir.join("decode_seq_for_verify.json"))
                .expect("verify json");
            write!(
                f,
                "{{\n  \"variant\": \"{}\",\n  \"prompt\": [{}],\n  \"tokens_run0\": [{}]\n}}\n",
                spec.variant_tag,
                prompt
                    .iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
                run_tokens[0]
                    .iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            )
            .expect("write verify json");
            println!(
                "banked {} (CPU verifier input)",
                out_dir.join("decode_seq_for_verify.json").display()
            );
        }
    }

    // ---- gate (d): determinism
    let det_tokens = run_tokens[0] == run_tokens[1];
    let det_sha = run_shas[0] == run_shas[1];
    println!(
        "GATE (d) determinism: tokens {} | logits streams {} (sha {})",
        if det_tokens { "IDENTICAL" } else { "DIVERGENT" },
        if det_sha {
            "BYTE-IDENTICAL"
        } else {
            "DIVERGENT"
        },
        run_shas[0]
    );

    // ---- gate (b) RAW: decode vs the lane-4 banked run (corrected verdict needs the
    // CPU rows — dsv4-decode-oracle-check; here: raw counts + first divergence)
    let n_cmp = lane4_tokens.len().min(run_tokens[0].len()).min(160);
    let agree = (0..n_cmp)
        .filter(|&i| run_tokens[0][i] == lane4_tokens[i])
        .count();
    let first_div = (0..n_cmp).find(|&i| run_tokens[0][i] != lane4_tokens[i]);
    println!(
        "GATE (b) RAW greedy identity vs lane-4 banked: {agree}/{n_cmp} agree (positions after a first divergence are trajectory-shifted, not errors); first divergence: {}",
        first_div.map_or("NONE".to_string(), |i| format!(
            "step {i} (decode {} vs lane4 {})",
            run_tokens[0][i], lane4_tokens[i]
        ))
    );

    // ---- gates (a1)+(c): re-prefill compares at the checkpoints (corrected policy)
    let full_ids: Vec<u32> = {
        let mut v = prompt.clone();
        v.extend_from_slice(&run_tokens[0]);
        v
    };
    let ev = (2.0f64 * 129280f64.ln()).sqrt();
    let mut gate_a1 = true;
    let mut n_bitexact = 0usize;
    let mut worst: (f64, usize) = (0.0, 0);
    let mut dec_blob: Vec<u8> = Vec::new();
    let mut rep_blob: Vec<u8> = Vec::new();
    println!(
        "GATE (a1) decode vs re-prefill (pair bounds: thr = 2·C·√(2 ln n)·absmax_ref, bands = 2·3·√2·C·|boundary|; class C = {:.4e} [{}]):",
        head_coeff(),
        if expert_arm_native() {
            "NATIVE expert arm"
        } else {
            "bf16-dequant arm"
        }
    );
    println!(
        "| s | max-abs | thr | m-floor | argmax d/r | in-band | top5 ovl/viol | top20 ovl/viol | verdict |"
    );
    let mut prev: Option<(usize, Vec<f32>)> = None;
    for (s_now, dec) in &ckpt_decode {
        let ref_out = gpu
            .forward(&full_ids[0..*s_now], None, None)
            .expect("re-prefill")
            .expect("logits");
        let rl = ref_out.logits;
        // m-floor context: the SAME previous row under this m (consecutive checkpoints)
        let floor = match &prev {
            Some((ps, pl)) if *ps + 1 == *s_now => {
                let again = gpu
                    .trunk_logits_row(&ref_out.h_last, *s_now, ps - 1)
                    .expect("floor row");
                let mut fa = 0f64;
                for (&a, &b) in pl.iter().zip(&again) {
                    fa = fa.max((a as f64 - b as f64).abs());
                }
                format!("{fa:.2e}")
            }
            _ => "-".into(),
        };
        let mut max_abs = 0f64;
        let mut absmax_ref = 0f32;
        for (&g, &r) in dec.iter().zip(&rl) {
            max_abs = max_abs.max((g as f64 - r as f64).abs());
            absmax_ref = absmax_ref.max(r.abs());
        }
        let thr = 2.0 * head_coeff() * ev * absmax_ref as f64;
        let am_d = argmax(dec);
        let am_r = argmax(&rl);
        let band_of = |x: f64| 2.0 * 3.0 * 2f64.sqrt() * head_coeff() * x.abs();
        let argmax_ok = if am_d == am_r {
            true
        } else {
            // pair band on the reference row's margin between the two candidates
            let margin = rl[am_r as usize] as f64 - rl[am_d as usize] as f64;
            margin <= band_of(rl[am_r as usize] as f64)
        };
        let (ov5, viol5) =
            band_violations(dec, &rl, 5, band_of(rl[top_ids(&rl, 5)[4] as usize] as f64));
        let (ov20, viol20) = band_violations(
            dec,
            &rl,
            20,
            band_of(rl[top_ids(&rl, 20)[19] as usize] as f64),
        );
        let ok = max_abs <= thr && argmax_ok && viol5 == 0 && viol20 == 0;
        if max_abs == 0.0 {
            n_bitexact += 1;
        }
        if max_abs > worst.0 {
            worst = (max_abs, *s_now);
        }
        if !ok {
            gate_a1 = false;
        }
        println!(
            "| {s_now} | {max_abs:.3e} | {thr:.3e} | {floor} | {am_d}/{am_r} | {} | {ov5}/5 v{viol5} | {ov20}/20 v{viol20} | {} |",
            if am_d == am_r {
                "same".into()
            } else {
                format!("flip({})", if argmax_ok { "in-band" } else { "OUT" })
            },
            if ok { "PASS" } else { "FAIL" }
        );
        for v in dec {
            dec_blob.extend_from_slice(&v.to_le_bytes());
        }
        for v in &rl {
            rep_blob.extend_from_slice(&v.to_le_bytes());
        }
        prev = Some((*s_now, rl));
    }
    std::fs::write(out_dir.join("decode_ckpt_logits.bin"), &dec_blob).expect("bank dec");
    std::fs::write(out_dir.join("reprefill_ckpt_logits.bin"), &rep_blob).expect("bank rep");
    println!(
        "checkpoint rows banked: decode_ckpt_logits.bin sha256 {} | reprefill_ckpt_logits.bin sha256 {}",
        sha256_hex(&dec_blob),
        sha256_hex(&rep_blob)
    );
    println!(
        "step-equivalence summary: {}/{} checkpoints bit-exact; worst max-abs {:.3e} at s={}",
        n_bitexact,
        ckpt_decode.len(),
        worst.0,
        worst.1,
    );
    // gate (c): the lane-6 LONG probe (>= 6 checkpoints at s >= 1024, saturation) when
    // the run reaches those lengths; otherwise the lane-7 SHORT-probe protocol (banked
    // in RECEIPTS.md "Lane 7" gate (d) BEFORE any run): >= 40 steps spanning fine-block
    // completions, coarse block 0 (s=127/128), the window wrap (s=129) and coarse
    // block 1 (s=256/257) — the s-dependent cache machinery is arm-independent and
    // stays covered by the lane-6 full-depth PASS.
    let late = ckpt_decode.iter().filter(|(s, _)| *s >= 1024).count();
    let gate_c = if s_max >= 1024 {
        println!("GATE (c) long-probe: {late} checkpoints at s >= 1024 (need >= 6)");
        late >= 6
    } else {
        let have = |s: usize| ckpt_decode.iter().any(|(cs, _)| *cs == s);
        let boundaries_ok =
            have(127) && have(128) && have(129) && have(256) && have(257) && n_new >= 40;
        println!(
            "GATE (c) short-probe (lane-7 protocol): n_new {n_new} >= 40, boundary checkpoints 127/128/129/256/257 {} (s >= 1024 and top-512 saturation NOT reachable at this length — covered by the lane-6 full-depth run; the caches are arm-independent)",
            if boundaries_ok {
                "ALL PRESENT"
            } else {
                "MISSING"
            }
        );
        boundaries_ok
    };

    // gate (e)
    let gate_e = cache_alloc_measured == expected_bytes;
    println!(
        "GATE (e) cache-alloc vs design math: measured {:?} vs formula {:?} -> {}",
        cache_alloc_measured,
        expected_bytes,
        if gate_e { "MATCH" } else { "MISMATCH" }
    );

    // informational latency shape
    let mean = |v: &Vec<f64>| {
        if v.is_empty() {
            f64::NAN
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    };
    println!(
        "informational (single-run, NOT perf claims): decode {:.0} ms/step at s in [190,210] (n={}), {:.0} ms/step at s in [1014,1034] (n={}) | lane-4 re-prefill was ~1100 ms/step at s in [190,210]",
        mean(&timing_200),
        timing_200.len(),
        mean(&timing_1024),
        timing_1024.len()
    );

    // bank the record (incl. prompt + checkpoints for the oracle-check binary)
    let mut f = std::fs::File::create(out_dir.join("decode_gate.json")).expect("json");
    write!(
        f,
        "{{\n  \"variant\": \"{}\",\n  \"n_new\": {},\n  \"prompt\": [{}],\n  \"tokens_run0\": [{}],\n  \"checkpoints\": [{}],\n  \"stream_sha256\": [\"{}\", \"{}\"],\n  \"gate_a1\": {gate_a1},\n  \"gate_b_raw_agree\": {agree},\n  \"gate_b_first_div\": {},\n  \"gate_c\": {gate_c},\n  \"gate_d\": {},\n  \"gate_e\": {gate_e},\n  \"n_checkpoints\": {},\n  \"n_bitexact\": {n_bitexact},\n  \"worst_max_abs\": {},\n  \"saturation_reached\": {saturation_reached}\n}}\n",
        spec.variant_tag,
        n_new,
        prompt
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(","),
        run_tokens[0]
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(","),
        ckpt_decode
            .iter()
            .map(|(s, _)| s.to_string())
            .collect::<Vec<_>>()
            .join(","),
        run_shas[0],
        run_shas[1],
        first_div.map(|i| i.to_string()).unwrap_or("null".into()),
        det_tokens && det_sha,
        ckpt_decode.len(),
        worst.0,
    )
    .expect("write json");

    let all = gate_a1 && gate_c && det_tokens && det_sha && gate_e;
    println!(
        "DSV4 GPU DECODE GATE: {} (a1 {} | b-raw {agree}/{n_cmp} [verdict via oracle-check] | c {} | d {} | e {}) | total {:.0}s",
        if all { "PASS (a1/c/d/e)" } else { "FAIL" },
        gate_a1,
        gate_c,
        det_tokens && det_sha,
        gate_e,
        t0.elapsed().as_secs_f64()
    );
    std::process::exit(if all { 0 } else { 1 });
}
