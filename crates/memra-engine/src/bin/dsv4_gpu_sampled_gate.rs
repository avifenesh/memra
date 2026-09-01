//! dsv4-gpu-sampled-gate: the SAMPLED spec==plain identity gate (ds4f rung 2, slice 1
//! — it5 item 8's "no sampled dsv4 spec path", finally answerable).
//!
//! Law: at temperature > 0, the drafted path must emit EXACTLY the plain sampled
//! stream at the same seed. That identity is structural here — the sampler is a pure
//! function of (logits row, absolute position, seed) with POSITION-KEYED draws, and
//! the batched verify's rows are bit-exact against the sequential step's (the it3
//! gate (c) proof) — and this gate is where the construction meets the hardware:
//! any misalignment (row/position off-by-one, RNG stream coupling, rollback leak)
//! breaks identity loudly on real prompts.
//!
//! Cells, one process, one load:
//!   1. per (seed x prompt): plain-sampled stream vs spec-sampled stream — IDENTICAL
//!      or the gate fails with the first divergence index and both tokens named;
//!   2. determinism: the spec run repeats byte-identically (same seed, fresh states);
//!   3. refusals: temperature 0 through the sampled path refuses BY NAME (pre-GPU);
//!   4. report: accepts/round + tokens/round at serving temperature — the observable
//!      rung 2 exists to unlock (correctness receipt here, never a speed claim).
//!
//! Sampling knobs (vendor posture defaults): MEMRA_DSV4_SAMPLE_TEMP (1.0),
//! MEMRA_DSV4_SAMPLE_TOPP (0.95), MEMRA_DSV4_SAMPLE_TOPK (0 = off).
//! Penalties are slice 2 and NOT claimed by this gate.
//!
//! Usage: dsv4-gpu-sampled-gate <model-dir> <prompts.json> <out-dir> [n_new] \
//!          [seeds,csv] [n_prompts] [dev0,dev1]

use memra_engine::dsv4_gpu::{
    Dsv4Gpu, Dsv4PenaltyCfg, Dsv4SampleCfg, Dsv4Vt, dsv4_penalize_row, dsv4_sample_row,
};
use memra_gguf::dsv4_forward::ActQuantVariant;
use std::io::Write;
use std::path::Path;

fn load_prompts(path: &Path) -> Vec<(String, Vec<u32>)> {
    let s = std::fs::read_to_string(path).expect("read prompts");
    let mut out = Vec::new();
    let mut rest = s.as_str();
    while let Some(i) = rest.find("\"pool\"") {
        rest = &rest[i + 6..];
        let q0 = rest.find('"').expect("pool open quote");
        let after = &rest[q0 + 1..];
        let q1 = after.find('"').expect("pool close quote");
        let pool = after[..q1].to_string();
        rest = &after[q1 + 1..];
        let j = rest.find("\"ids\"").expect("ids key");
        rest = &rest[j + 5..];
        let b0 = rest.find('[').expect("ids open");
        let b1 = rest.find(']').expect("ids close");
        let ids: Vec<u32> = rest[b0 + 1..b1]
            .split(',')
            .filter_map(|t| t.trim().parse::<u32>().ok())
            .collect();
        rest = &rest[b1 + 1..];
        assert!(!ids.is_empty(), "empty prompt ids for pool {pool}");
        out.push((pool, ids));
    }
    assert!(!out.is_empty(), "no prompts parsed");
    out
}

fn env_f32(name: &str, default: f32) -> f32 {
    match std::env::var(name) {
        Err(_) => default,
        Ok(s) => s
            .trim()
            .parse()
            .unwrap_or_else(|_| panic!("{name} '{s}' is not a float")),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: dsv4-gpu-sampled-gate <model-dir> <prompts.json> <out-dir> [n_new] \
             [seeds,csv] [n_prompts] [dev0,dev1]"
        );
        std::process::exit(2);
    }
    let t0 = std::time::Instant::now();
    let dir = Path::new(&args[1]);
    let mut prompts = load_prompts(Path::new(&args[2]));
    let out_dir = Path::new(&args[3]);
    std::fs::create_dir_all(out_dir).expect("mkdir out");
    let n_new: usize = args.get(4).map(|x| x.parse().expect("n_new")).unwrap_or(64);
    let seeds: Vec<u64> = args
        .get(5)
        .map(|s| {
            s.split(',')
                .map(|x| x.trim().parse().expect("seed"))
                .collect()
        })
        .unwrap_or_else(|| vec![20260822, 7, 424242]);
    let n_prompts: usize = args
        .get(6)
        .map(|x| x.parse().expect("n_prompts"))
        .unwrap_or(8);
    let devices: Vec<usize> = args
        .get(7)
        .map(|s| s.split(',').map(|x| x.parse().expect("device")).collect())
        .unwrap_or_else(|| vec![0, 1]);
    prompts.truncate(n_prompts);
    if std::env::var("MEMRA_DSV4_DRAFTER").as_deref() != Ok("dspark") {
        eprintln!("REFUSE: this gate requires MEMRA_DSV4_DRAFTER=dspark");
        std::process::exit(2);
    }
    if std::env::var("MEMRA_DSV4_DECODE_PATH").as_deref() != Ok("device") {
        eprintln!("REFUSE: this gate requires MEMRA_DSV4_DECODE_PATH=device");
        std::process::exit(2);
    }
    let base = Dsv4SampleCfg {
        temperature: env_f32("MEMRA_DSV4_SAMPLE_TEMP", 1.0),
        top_p: env_f32("MEMRA_DSV4_SAMPLE_TOPP", 0.95),
        top_k: std::env::var("MEMRA_DSV4_SAMPLE_TOPK")
            .ok()
            .map(|s| s.trim().parse().expect("MEMRA_DSV4_SAMPLE_TOPK"))
            .unwrap_or(0),
        seed: 0,
    };

    // cell 3 first — refusals are pre-GPU and must never cost a load
    {
        let bad = Dsv4SampleCfg {
            temperature: 0.0,
            ..base
        };
        match dsv4_sample_row(&[0.0, 1.0], 0, &bad) {
            Err(e) if e.contains("temperature") => {
                println!("refusal cell: temperature 0 refuses by name — OK")
            }
            other => {
                eprintln!("REFUSAL CELL FAIL: t=0 gave {other:?}");
                std::process::exit(1);
            }
        }
    }

    println!(
        "dsv4-gpu-sampled-gate | model {} | {} prompts x {} seeds x {n_new} tok | temp {} \
         top_p {} top_k {} | SAMPLED spec==plain identity is the verdict",
        dir.display(),
        prompts.len(),
        seeds.len(),
        base.temperature,
        base.top_p,
        base.top_k
    );
    let variant = ActQuantVariant::ClampOnly;
    let max_p = prompts.iter().map(|(_, i)| i.len()).max().unwrap();
    let max_seq = (max_p + n_new + 96).max(256);
    let gpu = Dsv4Gpu::load(dir, &devices, variant, max_seq).expect("load");
    println!(
        "loaded: split at layer {}, t={:.0}s",
        gpu.split_at,
        t0.elapsed().as_secs_f64()
    );

    let mut fails: Vec<String> = Vec::new();
    let mut det_fails = 0usize;
    let mut rounds_total = 0usize;
    let mut accepts_total = 0usize;
    let mut emitted_total = 0usize;
    let mut rows: Vec<String> = Vec::new();

    for &seed in &seeds {
        let cfg = Dsv4SampleCfg { seed, ..base };
        for (pi, (pool, ids)) in prompts.iter().enumerate() {
            let p0 = ids.len();
            // plain sampled stream (sequential decode steps, position-keyed draws)
            let plain: Vec<u32> = {
                let mut state = gpu.alloc_decode_state().expect("alloc state");
                let pre = gpu.prefill_with_cache(ids, &mut state).expect("prefill");
                let mut t = dsv4_sample_row(&pre.logits, p0, &cfg).expect("draw");
                let mut toks = Vec::with_capacity(n_new);
                for step in 0..n_new {
                    toks.push(t);
                    if step + 1 == n_new {
                        break;
                    }
                    let row = gpu.decode_step(t, &mut state).expect("plain step");
                    t = dsv4_sample_row(&row, p0 + step + 1, &cfg).expect("draw");
                }
                toks
            };
            // spec sampled stream x2 (fresh states each — determinism is cell 2)
            let mut spec_runs = Vec::new();
            for _ in 0..2 {
                let mut state = gpu.alloc_decode_state().expect("alloc state");
                let mut dstate = gpu.dspark_alloc_state().expect("alloc dspark");
                let mut vstate = gpu.alloc_verify_state().expect("alloc verify");
                let out = gpu
                    .spec_sampled_batched_policy(
                        ids,
                        n_new,
                        &mut state,
                        &mut dstate,
                        &mut vstate,
                        usize::MAX,
                        Dsv4Vt::Off,
                        &cfg,
                    )
                    .expect("spec sampled run");
                spec_runs.push(out);
            }
            if spec_runs[0].tokens != spec_runs[1].tokens {
                det_fails += 1;
                fails.push(format!(
                    "DETERMINISM FAIL seed {seed} prompt {pi} ({pool}): two spec runs differ"
                ));
            }
            let spec = &spec_runs[0];
            if spec.tokens != plain {
                let first = spec
                    .tokens
                    .iter()
                    .zip(&plain)
                    .position(|(a, b)| a != b)
                    .unwrap_or(plain.len().min(spec.tokens.len()));
                fails.push(format!(
                    "IDENTITY FAIL seed {seed} prompt {pi} ({pool}): spec != plain at \
                     generated index {first} (spec {:?} vs plain {:?})",
                    spec.tokens.get(first),
                    plain.get(first)
                ));
            }
            let r = spec.rounds.len();
            let a: usize = spec.rounds.iter().map(|x| x.accepts).sum();
            rounds_total += r;
            accepts_total += a;
            emitted_total += spec.tokens.len();
            let row = format!(
                "seed {seed} p{pi:02} [{pool}]: identity {} | det {} | rounds {r} accepted {a} \
                 ({:.3}/round, {:.3} tok/round)",
                if spec.tokens == plain { "PASS" } else { "FAIL" },
                if spec_runs[0].tokens == spec_runs[1].tokens {
                    "PASS"
                } else {
                    "FAIL"
                },
                a as f64 / r.max(1) as f64,
                spec.tokens.len() as f64 / r.max(1) as f64
            );
            println!("{row}");
            rows.push(row);
        }
    }

    // ---- rung-2 slice 2: PENALIZED sampled identity (fixed coefficients, 1 seed) ----
    let pen = Dsv4PenaltyCfg {
        last_n: 256,
        repeat: 1.3,
        freq: 0.2,
        present: 0.2,
    };
    let pen_seed = 20260822u64;
    let pen_cfg = Dsv4SampleCfg {
        seed: pen_seed,
        ..base
    };
    for (pi, (pool, ids)) in prompts.iter().take(4).enumerate() {
        let p0 = ids.len();
        let plain: Vec<u32> = {
            let mut state = gpu.alloc_decode_state().expect("alloc state");
            let pre = gpu.prefill_with_cache(ids, &mut state).expect("prefill");
            let mut window: Vec<u32> = ids.clone();
            let mut row0 = pre.logits.clone();
            dsv4_penalize_row(&mut row0, &window, &pen);
            let mut t = dsv4_sample_row(&row0, p0, &pen_cfg).expect("draw");
            let mut toks = Vec::with_capacity(n_new);
            for step in 0..n_new {
                toks.push(t);
                window.push(t);
                if step + 1 == n_new {
                    break;
                }
                let mut row = gpu.decode_step(t, &mut state).expect("plain step");
                dsv4_penalize_row(&mut row, &window, &pen);
                t = dsv4_sample_row(&row, p0 + step + 1, &pen_cfg).expect("draw");
            }
            toks
        };
        let mut runs = Vec::new();
        for _ in 0..2 {
            let mut state = gpu.alloc_decode_state().expect("alloc state");
            let mut dstate = gpu.dspark_alloc_state().expect("alloc dspark");
            let mut vstate = gpu.alloc_verify_state().expect("alloc verify");
            let out = gpu
                .spec_sampled_batched_pen(
                    ids,
                    n_new,
                    &mut state,
                    &mut dstate,
                    &mut vstate,
                    usize::MAX,
                    Dsv4Vt::Off,
                    &pen_cfg,
                    Some(&pen),
                    None,
                )
                .expect("spec penalized run");
            runs.push(out.tokens);
        }
        if runs[0] != runs[1] {
            fails.push(format!("PEN DETERMINISM FAIL prompt {pi} ({pool})"));
        }
        if runs[0] != plain {
            let first = runs[0]
                .iter()
                .zip(&plain)
                .position(|(a, b)| a != b)
                .unwrap_or(plain.len().min(runs[0].len()));
            fails.push(format!(
                "PEN IDENTITY FAIL prompt {pi} ({pool}): spec != plain at index {first} \
                 (spec {:?} vs plain {:?})",
                runs[0].get(first),
                plain.get(first)
            ));
        }
        let row = format!(
            "PEN seed {pen_seed} p{pi:02} [{pool}]: identity {}",
            if runs[0] == plain { "PASS" } else { "FAIL" }
        );
        println!("{row}");
        rows.push(row);
    }

    println!(
        "\n=== SAMPLED SPEC GATE | serving-temp accept (CORRECTNESS observable, never a \
         speed claim): {:.4}/round accepted, {:.4} tok/round over {} rounds ===",
        accepts_total as f64 / rounds_total.max(1) as f64,
        emitted_total as f64 / rounds_total.max(1) as f64,
        rounds_total
    );
    let mut f = std::fs::File::create(out_dir.join("sampled_gate.txt")).expect("out");
    for r in &rows {
        writeln!(f, "{r}").expect("w");
    }
    for r in &fails {
        writeln!(f, "{r}").expect("w");
    }
    if fails.is_empty() {
        println!(
            "GPU SAMPLED GATE [PASS] — identity + determinism on every (seed, prompt); \
             t={:.0}s",
            t0.elapsed().as_secs_f64()
        );
    } else {
        for l in &fails {
            eprintln!("{l}");
        }
        eprintln!(
            "GPU SAMPLED GATE [FAIL] — {} failure(s), {det_fails} determinism",
            fails.len()
        );
        std::process::exit(1);
    }
}
