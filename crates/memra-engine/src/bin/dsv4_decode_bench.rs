//! dsv4-decode-bench: lane-8 decode perf instrument (RECEIPTS.md "Lane 8").
//!
//! Prefills the fixture 32-token prompt with cache population, then greedy-decodes
//! `n_new` steps through the incremental decode path, recording per-step wall time.
//! Reports ms/step MEDIANS over windows centered on the target sequence lengths
//! (s∈[t−10, t+10]) plus min/max spreads. Banks tokens + the sha256 of the full
//! logits stream so byte-identity across rungs/arms is checkable from the same run.
//!
//! ONE run of this binary is a single observation — perf CLAIMS come only from the
//! interleaved A/B protocol (alternating seam values, ×5, medians + spreads), driven
//! by a wrapper script. Everything a single invocation prints is informational.
//!
//! Seams (read at load, printed): MEMRA_DSV4_EXPERT_ARM (lane 7),
//! MEMRA_DSV4_DECODE_PATH (lane 8, once it exists).
//!
//! Usage: dsv4-decode-bench <model-dir> <fixtures.json> <out.json> [n_new] [dev0,dev1]

use memra_engine::dsv4_gpu::Dsv4Gpu;
use memra_gguf::dsv4_forward::FixtureSpec;
use std::io::Write;
use std::path::Path;

fn vram_line(gpu: &Dsv4Gpu, tag: &str) {
    let line = match gpu.vram_report() {
        Ok(rows) => rows
            .iter()
            .map(|(dev, free, total, resident)| {
                format!(
                    "dev{dev} used {:.2}/{:.2} GiB (resident {:.2})",
                    (*total - *free) as f64 / 2f64.powi(30),
                    *total as f64 / 2f64.powi(30),
                    *resident as f64 / 2f64.powi(30)
                )
            })
            .collect::<Vec<_>>()
            .join(" | "),
        Err(err) => format!("vram report failed: {err}"),
    };
    println!("[vram {tag}] {line}");
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

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

fn median_us(mut v: Vec<u64>) -> f64 {
    assert!(!v.is_empty());
    v.sort_unstable();
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2] as f64
    } else {
        (v[n / 2 - 1] + v[n / 2]) as f64 / 2.0
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: dsv4-decode-bench <model-dir> <fixtures.json> <out.json> [n_new] [dev0,dev1]"
        );
        std::process::exit(2);
    }
    let t0 = std::time::Instant::now();
    let dir = Path::new(&args[1]);
    let spec = FixtureSpec::load(Path::new(&args[2]));
    let out_path = Path::new(&args[3]);
    let n_new: usize = args
        .get(4)
        .map(|x| x.parse().expect("n_new"))
        .unwrap_or(1024);
    let devices: Vec<usize> = args
        .get(5)
        .map(|s| s.split(',').map(|x| x.parse().expect("device")).collect())
        .unwrap_or_else(|| vec![0, 1]);
    let arm = std::env::var("MEMRA_DSV4_EXPERT_ARM").unwrap_or_default();
    let path_seam = std::env::var("MEMRA_DSV4_DECODE_PATH").unwrap_or_default();
    println!(
        "dsv4-decode-bench | model {} | variant {} | n_new {n_new} | devices {devices:?} | \
         MEMRA_DSV4_EXPERT_ARM='{arm}' MEMRA_DSV4_DECODE_PATH='{path_seam}'",
        dir.display(),
        spec.variant_tag
    );

    let prompt = spec.tokens_32.clone();
    let max_seq = (prompt.len() + n_new + 96).max(256);
    let gpu = Dsv4Gpu::load(dir, &devices, spec.variant, max_seq).expect("load");
    println!(
        "loaded: split at layer {}, t={:.0}s",
        gpu.split_at,
        t0.elapsed().as_secs_f64()
    );
    vram_line(&gpu, "post-load");

    // MEMRA_DSV4_BENCH_DRAFTED=1: the iteration-3 DRAFTED arm — DSpark proposal +
    // BATCHED T=k+1 device verify + §3.1 commit/rollback, timed PER ROUND and attributed
    // to the tokens the round emitted. Requires MEMRA_DSV4_DRAFTER=dspark and the device
    // decode path. Everything a single invocation prints is still informational; the
    // claim comes from the interleaved A/B wrapper.
    if std::env::var("MEMRA_DSV4_BENCH_DRAFTED").as_deref() == Ok("1") {
        run_drafted(
            &gpu,
            &prompt,
            n_new,
            out_path,
            &spec.variant_tag,
            &arm,
            &path_seam,
            t0,
        );
        return;
    }

    let mut state = gpu.alloc_decode_state().expect("alloc state");
    vram_line(&gpu, "post-alloc");
    let pre_t0 = std::time::Instant::now();
    let pre = gpu
        .prefill_with_cache(&prompt, &mut state)
        .expect("prefill");
    let prefill_ms = pre_t0.elapsed().as_secs_f64() * 1e3;
    println!(
        "prefill {} tokens: {prefill_ms:.1} ms (informational)",
        prompt.len()
    );

    // MEMRA_DSV4_BENCH_PROFILE=1: bracket steps [16, 48) with cudaProfilerStart/Stop so
    // `nsys profile -c cudaProfilerApi` captures ONLY steady-state decode steps (no load,
    // no prefill). Profiling runs are rung-0 instruments, never A/B observations.
    let profile_bracket = std::env::var("MEMRA_DSV4_BENCH_PROFILE").as_deref() == Ok("1");

    // MEMRA_DSV4_BENCH_GREEDY=1: the serving shape — decode_step_greedy returns only
    // the next token (device argmax on the device path); no logits stream is banked
    // (its byte-identity witness is the full-logits mode of the same seams).
    let greedy_mode = std::env::var("MEMRA_DSV4_BENCH_GREEDY").as_deref() == Ok("1");

    let mut tok = argmax(&pre.logits);
    let mut tokens: Vec<u32> = vec![tok];
    let mut step_us: Vec<u64> = Vec::with_capacity(n_new);
    let mut logits_blob: Vec<u8> =
        Vec::with_capacity(if greedy_mode { 0 } else { n_new * 129280 * 4 });
    let run_t0 = std::time::Instant::now();
    for step in 0..n_new {
        if profile_bracket && step == 16 {
            cudarc::driver::safe::profiler_start().expect("profiler_start");
        }
        if profile_bracket && step == 48 {
            cudarc::driver::safe::profiler_stop().expect("profiler_stop");
        }
        let st = std::time::Instant::now();
        if greedy_mode {
            tok = gpu
                .decode_step_greedy(tok, &mut state)
                .expect("decode_step_greedy");
            step_us.push(st.elapsed().as_micros() as u64);
        } else {
            let logits = gpu.decode_step(tok, &mut state).expect("decode_step");
            step_us.push(st.elapsed().as_micros() as u64);
            tok = argmax(&logits);
            for v in &logits {
                logits_blob.extend_from_slice(&v.to_le_bytes());
            }
        }
        tokens.push(tok);
        if step % 128 == 0 || step + 1 == n_new {
            println!(
                "step {step}: s={} tok {tok} {:.1} ms t={:.1}s",
                state.pos,
                step_us.last().copied().unwrap_or(0) as f64 / 1e3,
                run_t0.elapsed().as_secs_f64()
            );
        }
    }
    vram_line(&gpu, "post-run");
    let sha = if greedy_mode {
        format!(
            "tokens-only:{}",
            sha256_hex(
                &tokens
                    .iter()
                    .flat_map(|t| t.to_le_bytes())
                    .collect::<Vec<u8>>()
            )
        )
    } else {
        sha256_hex(&logits_blob)
    };
    println!(
        "decoded {n_new} steps in {:.1}s | stream sha256 {sha}",
        run_t0.elapsed().as_secs_f64()
    );

    // windows: step i runs at s = prompt.len() + i (positions consumed before the step)
    let s0 = prompt.len();
    let mut window_lines: Vec<String> = Vec::new();
    let mut window_json: Vec<String> = Vec::new();
    for target in [200usize, 512, 1024, 2048, 4096, 8192] {
        let lo = target.saturating_sub(10);
        let hi = target + 10;
        let in_win: Vec<u64> = step_us
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                let s = s0 + i;
                s >= lo && s <= hi
            })
            .map(|(_, &u)| u)
            .collect();
        if in_win.len() < 11 {
            continue;
        }
        let med = median_us(in_win.clone());
        let mn = *in_win.iter().min().unwrap();
        let mx = *in_win.iter().max().unwrap();
        let mean = in_win.iter().sum::<u64>() as f64 / in_win.len() as f64;
        window_lines.push(format!(
            "s≈{target}: median {:.1} ms/step (min {:.1} / max {:.1}, n={}) = {:.1} tok/s \
             [mean {:.1} ms = {:.1} tok/s]",
            med / 1e3,
            mn as f64 / 1e3,
            mx as f64 / 1e3,
            in_win.len(),
            1e6 / med,
            mean / 1e3,
            1e6 / mean
        ));
        window_json.push(format!(
            "{{\"s\": {target}, \"median_us\": {med:.1}, \"mean_us\": {mean:.1}, \"min_us\": {mn}, \"max_us\": {mx}, \"n\": {}}}",
            in_win.len()
        ));
    }
    for l in &window_lines {
        println!("[window] {l}");
    }

    // WIDE bands — the same generated-token ranges the drafted arm reports, so the two
    // arms are read side by side on a statistic with enough samples to be stable.
    let mut band_json: Vec<String> = Vec::new();
    for (lo, hi) in [
        (0usize, 200usize),
        (200, 512),
        (512, 1024),
        (1024, 2048),
        (2048, 4096),
        (4096, usize::MAX),
    ] {
        let hi_i = hi.min(step_us.len());
        if lo >= hi_i || hi_i - lo < 20 {
            continue;
        }
        let seg = &step_us[lo..hi_i];
        let mean = seg.iter().sum::<u64>() as f64 / seg.len() as f64;
        let hi_s = if hi == usize::MAX {
            "end".to_string()
        } else {
            hi.to_string()
        };
        println!(
            "[band] tokens [{lo},{hi_s}): {:.2} ms/token (n={}) = {:.1} tok/s",
            mean / 1e3,
            seg.len(),
            1e6 / mean
        );
        band_json.push(format!(
            "{{\"lo\": {lo}, \"hi\": \"{hi_s}\", \"mean_us\": {mean:.1}, \"n\": {}}}",
            seg.len()
        ));
    }

    let mut f = std::fs::File::create(out_path).expect("json");
    write!(
        f,
        "{{\n  \"variant\": \"{}\",\n  \"expert_arm\": \"{arm}\",\n  \"decode_path\": \"{path_seam}\",\n  \
         \"n_new\": {n_new},\n  \"prefill_ms\": {prefill_ms:.1},\n  \"logits_sha256\": \"{sha}\",\n  \
         \"tokens\": [{}],\n  \"windows\": [{}],\n  \"bands\": [{}],\n  \"step_us\": [{}]\n}}\n",
        spec.variant_tag,
        tokens
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(","),
        window_json.join(", "),
        band_json.join(", "),
        step_us
            .iter()
            .map(|u| u.to_string())
            .collect::<Vec<_>>()
            .join(",")
    )
    .expect("write json");
    println!(
        "banked {} | total elapsed {:.0}s | single-run: informational, NOT a perf claim",
        out_path.display(),
        t0.elapsed().as_secs_f64()
    );
}

/// The iteration-3 DRAFTED arm: DSpark proposal + batched T=k+1 device verify + §3.1
/// commit/rollback. A round emits `1 + accepts` tokens for one round of wall time, so the
/// per-token cost is `round_us / emitted` and a window's ms/token is the SUM of the
/// shares over the tokens whose positions fall in the window divided by their count —
/// i.e. total time over total tokens, which is the only statistic a throughput claim may
/// use here. (Medians of per-token shares would be meaningless when the shares differ
/// 6x by accept count, so the plain arm now prints its window MEAN too and the A/B
/// compares mean to mean.)
///
/// Acceptance numbers printed below are CORRECTNESS observables (dspark-q38 law); the
/// speed statement is the measured ms/token, never a projection from accept length.
#[allow(clippy::too_many_arguments)]
fn run_drafted(
    gpu: &Dsv4Gpu,
    prompt: &[u32],
    n_new: usize,
    out_path: &Path,
    variant_tag: &str,
    arm: &str,
    path_seam: &str,
    t0: std::time::Instant,
) {
    if gpu.verify_tmax() == 0 {
        eprintln!("REFUSE: MEMRA_DSV4_BENCH_DRAFTED=1 needs MEMRA_DSV4_DRAFTER=dspark");
        std::process::exit(2);
    }
    let p0 = prompt.len();
    let mut state = gpu.alloc_decode_state().expect("alloc state");
    let mut dstate = gpu.dspark_alloc_state().expect("alloc dspark state");
    let mut vstate = gpu.alloc_verify_state().expect("alloc verify state");
    vram_line(gpu, "post-alloc-drafted");
    println!(
        "drafted arm: verify tmax {} | verify-state bytes/dev {:?} | t={:.0}s",
        gpu.verify_tmax(),
        vstate.bytes,
        t0.elapsed().as_secs_f64()
    );
    let run_t0 = std::time::Instant::now();
    let out = gpu
        .spec_greedy_batched_with(prompt, n_new, &mut state, &mut dstate, &mut vstate)
        .expect("drafted run");
    let wall = run_t0.elapsed().as_secs_f64();
    vram_line(gpu, "post-run-drafted");

    // per-emitted-token shares, keyed on the position the token was emitted at
    let mut shares: Vec<(usize, f64)> = Vec::with_capacity(out.tokens.len());
    let mut g = 0usize;
    let mut round_us_all: Vec<u64> = Vec::new();
    let mut accepts_total = 0usize;
    let mut verified_total = 0usize;
    let mut t_batch_sum = 0usize;
    let mut t_batch_n = 0usize;
    for r in &out.rounds {
        let em = r.emitted.max(1);
        let share = r.round_us as f64 / em as f64;
        for j in 0..em {
            if g + j < out.tokens.len() {
                shares.push((p0 + g + j, share));
            }
        }
        g += em;
        round_us_all.push(r.round_us);
        accepts_total += r.accepts;
        verified_total += r.verified;
        if r.t_batch > 0 {
            t_batch_sum += r.t_batch;
            t_batch_n += 1;
        }
    }
    let rounds = out.rounds.len();

    // ---- SPS(B) profile inputs + per-position acceptance, printed and banked ----------
    // `round_us` at a pinned depth IS the SPS(B) datum (SPS(B) = 1e6 / mean round_us at
    // B = t_batch - 1 verified drafts). Rounds whose t_batch was cut by the n_new budget
    // (t_batch < t_cap) are EXCLUDED: their cost is not the cost of a depth-B round, and
    // including them would bias the tail of every sweep point downward.
    let steady: Vec<&memra_engine::dsv4_gpu::SpecRoundGpu> = out
        .rounds
        .iter()
        .filter(|r| r.t_batch > 0 && r.t_batch == r.t_cap)
        .collect();
    if !steady.is_empty() {
        let n = steady.len() as f64;
        let mean_us = steady.iter().map(|r| r.round_us as f64).sum::<f64>() / n;
        let mean_acc = steady.iter().map(|r| r.accepts as f64).sum::<f64>() / n;
        let mean_t = steady.iter().map(|r| r.t_batch as f64).sum::<f64>() / n;
        let mean_em = steady.iter().map(|r| r.emitted as f64).sum::<f64>() / n;
        // per-position conditional acceptance: of the rounds that reached slot j at all
        // (accepts >= j), what fraction accepted it (accepts > j)?
        let kmax = steady.iter().map(|r| r.t_batch - 1).max().unwrap_or(0);
        let mut pos: Vec<String> = Vec::new();
        for j in 0..kmax {
            let reached = steady.iter().filter(|r| r.accepts >= j).count();
            let took = steady.iter().filter(|r| r.accepts > j).count();
            if reached > 0 {
                pos.push(format!("p{}={:.3}", j + 1, took as f64 / reached as f64));
            }
        }
        println!(
            "[sps] steady rounds {} | mean T {:.3} | mean round {:.3} ms => SPS {:.2}/s | \
             mean accepts {:.4} | mean emitted (tau*) {:.4} | tok/s {:.2}",
            steady.len(),
            mean_t,
            mean_us / 1e3,
            1e6 / mean_us,
            mean_acc,
            mean_em,
            mean_em * 1e6 / mean_us
        );
        println!(
            "[sps] per-position conditional acceptance: {}",
            pos.join(" ")
        );
    }
    let rounds_side = out_path.with_extension("rounds.json");
    {
        let mut rf = std::fs::File::create(&rounds_side).expect("rounds json");
        writeln!(rf, "{{\"schema\": \"dsv4-spec-rounds-v1\", \"rounds\": [").expect("w");
        for (i, r) in out.rounds.iter().enumerate() {
            let conf = r
                .confidence
                .iter()
                .map(|c| format!("{c:.6}"))
                .collect::<Vec<_>>()
                .join(",");
            writeln!(
                rf,
                "  {{\"i\": {i}, \"start_pos\": {}, \"t_batch\": {}, \"t_cap\": {}, \
                 \"accepts\": {}, \"emitted\": {}, \"round_us\": {}, \"conf\": [{conf}]}}{}",
                r.start_pos,
                r.t_batch,
                r.t_cap,
                r.accepts,
                r.emitted,
                r.round_us,
                if i + 1 == out.rounds.len() { "" } else { "," }
            )
            .expect("w");
        }
        writeln!(rf, "]}}").expect("w");
    }
    println!("[sps] per-round records banked {}", rounds_side.display());

    println!(
        "drafted: {} tokens in {wall:.1}s | rounds {rounds} | accepted {accepts_total} \
         (mean {:.4}/round) | verified {verified_total} | mean T forwarded {:.4} | mean \
         tokens/round {:.4} | overall {:.2} ms/token = {:.1} tok/s",
        out.tokens.len(),
        accepts_total as f64 / rounds as f64,
        if t_batch_n > 0 {
            t_batch_sum as f64 / t_batch_n as f64
        } else {
            0.0
        },
        out.tokens.len() as f64 / rounds as f64,
        wall * 1e3 / out.tokens.len() as f64,
        out.tokens.len() as f64 / wall
    );
    println!(
        "*** acceptance is a CORRECTNESS observable, never a speed claim (dspark-q38 law); \
         the speed statement is the measured ms/token above ***"
    );
    let tok_sha = sha256_hex(
        &out.tokens
            .iter()
            .flat_map(|t| t.to_le_bytes())
            .collect::<Vec<u8>>(),
    );
    println!("stream sha256 tokens-only:{tok_sha}");

    let mut window_lines: Vec<String> = Vec::new();
    let mut window_json: Vec<String> = Vec::new();
    for target in [200usize, 512, 1024, 2048, 4096, 8192] {
        let lo = target.saturating_sub(10);
        let hi = target + 10;
        let in_win: Vec<f64> = shares
            .iter()
            .filter(|(s, _)| *s >= lo && *s <= hi)
            .map(|(_, u)| *u)
            .collect();
        if in_win.len() < 11 {
            continue;
        }
        let mean = in_win.iter().sum::<f64>() / in_win.len() as f64;
        let med = {
            let mut v = in_win.clone();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = v.len();
            if n % 2 == 1 {
                v[n / 2]
            } else {
                (v[n / 2 - 1] + v[n / 2]) / 2.0
            }
        };
        window_lines.push(format!(
            "s≈{target}: mean {:.1} ms/token (n={}) = {:.1} tok/s [median share {:.1} ms]",
            mean / 1e3,
            in_win.len(),
            1e6 / mean,
            med / 1e3
        ));
        window_json.push(format!(
            "{{\"s\": {target}, \"mean_us\": {mean:.1}, \"median_us\": {med:.1}, \"n\": {}}}",
            in_win.len()
        ));
    }
    for l in &window_lines {
        println!("[window] {l}");
    }

    // WIDE bands (total time / total tokens over a generated-token range). The 21-token
    // windows above are only ~4 drafted rounds wide, which is too few rounds to be a
    // stable per-context statistic for this arm; the bands are the ones a claim should
    // quote, and the plain arm prints the same bands so the two are read side by side.
    let mut band_lines: Vec<String> = Vec::new();
    let mut band_json: Vec<String> = Vec::new();
    for (lo, hi) in [
        (0usize, 200usize),
        (200, 512),
        (512, 1024),
        (1024, 2048),
        (2048, 4096),
        (4096, usize::MAX),
    ] {
        let hi_abs = if hi == usize::MAX {
            usize::MAX
        } else {
            p0 + hi
        };
        let seg: Vec<f64> = shares
            .iter()
            .filter(|(s, _)| *s >= p0 + lo && *s < hi_abs)
            .map(|(_, u)| *u)
            .collect();
        if seg.len() < 20 {
            continue;
        }
        let mean = seg.iter().sum::<f64>() / seg.len() as f64;
        let hi_s = if hi == usize::MAX {
            "end".to_string()
        } else {
            hi.to_string()
        };
        band_lines.push(format!(
            "tokens [{lo},{hi_s}): {:.2} ms/token (n={}) = {:.1} tok/s",
            mean / 1e3,
            seg.len(),
            1e6 / mean
        ));
        band_json.push(format!(
            "{{\"lo\": {lo}, \"hi\": \"{hi_s}\", \"mean_us\": {mean:.1}, \"n\": {}}}",
            seg.len()
        ));
    }
    for l in &band_lines {
        println!("[band] {l}");
    }

    let mut f = std::fs::File::create(out_path).expect("json");
    write!(
        f,
        "{{\n  \"arm\": \"drafted-batched\",\n  \"variant\": \"{variant_tag}\",\n  \
         \"expert_arm\": \"{arm}\",\n  \"decode_path\": \"{path_seam}\",\n  \
         \"n_new\": {n_new},\n  \"wall_s\": {wall:.3},\n  \"rounds\": {rounds},\n  \
         \"accepted\": {accepts_total},\n  \"verified\": {verified_total},\n  \
         \"tokens_sha256\": \"{tok_sha}\",\n  \"tokens\": [{}],\n  \"windows\": [{}],\n  \
         \"bands\": [{}],\n  \"round_us\": [{}],\n  \"round_emitted\": [{}]\n}}\n",
        out.tokens
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(","),
        window_json.join(", "),
        band_json.join(", "),
        round_us_all
            .iter()
            .map(|u| u.to_string())
            .collect::<Vec<_>>()
            .join(","),
        out.rounds
            .iter()
            .map(|r| r.emitted.to_string())
            .collect::<Vec<_>>()
            .join(",")
    )
    .expect("write json");
    println!(
        "banked {} | total elapsed {:.0}s | single-run: informational, NOT a perf claim",
        out_path.display(),
        t0.elapsed().as_secs_f64()
    );
}
