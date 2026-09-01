//! dsv4-gpu-greedy: DeepSeek-V4-Flash GPU greedy-continuation driver (lane 4, gate (b)
//! GPU side + gate (c) determinism/VRAM).
//!
//! Greedy-decodes `n_new` tokens from the fixture 32-token prompt by re-prefill per step
//! (the banked O(n²) bring-up rung — decode caching is a later lane), `runs` times, and
//! banks per-run token ids + every step's full logits row (f32 LE .bin) so the CPU
//! teacher-forcing verifier (memra-gguf dsv4-greedy-verify) can pin agreement and, on
//! divergence, both logit rows exist. Determinism = identical token sequences AND
//! byte-identical logits bins across runs.
//!
//! Usage: dsv4-gpu-greedy <model-dir> <fixtures.json> <out-dir> [n_new] [runs] [dev0,dev1]

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
    // tiny dependency-free FNV-free approach is not acceptable for receipts — reuse
    // sha2 via memra-gguf's re-export path is unavailable, so hash with sha2 directly.
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: dsv4-gpu-greedy <model-dir> <fixtures.json> <out-dir> [n_new] [runs] [dev0,dev1]"
        );
        std::process::exit(2);
    }
    let t0 = std::time::Instant::now();
    let dir = Path::new(&args[1]);
    let spec = FixtureSpec::load(Path::new(&args[2]));
    let out_dir = Path::new(&args[3]);
    std::fs::create_dir_all(out_dir).expect("mkdir out");
    let n_new: usize = args
        .get(4)
        .map(|x| x.parse().expect("n_new"))
        .unwrap_or(160);
    let runs: usize = args.get(5).map(|x| x.parse().expect("runs")).unwrap_or(2);
    let devices: Vec<usize> = args
        .get(6)
        .map(|s| s.split(',').map(|x| x.parse().expect("device")).collect())
        .unwrap_or_else(|| vec![0, 1]);
    println!(
        "dsv4-gpu-greedy | model {} | variant {} | n_new {n_new} | runs {runs} | devices {devices:?}",
        dir.display(),
        spec.variant_tag
    );

    let prompt = spec.tokens_32.clone();
    let max_seq = (prompt.len() + n_new + 32).max(256);
    let gpu = Dsv4Gpu::load(dir, &devices, spec.variant, max_seq).expect("load");
    println!(
        "loaded: split at layer {}, t={:.0}s",
        gpu.split_at,
        t0.elapsed().as_secs_f64()
    );
    vram_line(&gpu, "post-load");

    let mut run_tokens: Vec<Vec<u32>> = Vec::new();
    let mut run_shas: Vec<String> = Vec::new();
    for run in 0..runs {
        let mut ids = prompt.clone();
        let mut logits_blob: Vec<u8> = Vec::with_capacity(n_new * 129280 * 4);
        let mut new_tokens = Vec::with_capacity(n_new);
        let run_t0 = std::time::Instant::now();
        for step in 0..n_new {
            let logits = gpu
                .forward(&ids, None, None)
                .expect("forward")
                .expect("logits")
                .logits;
            let tok = argmax(&logits);
            for v in &logits {
                logits_blob.extend_from_slice(&v.to_le_bytes());
            }
            new_tokens.push(tok);
            ids.push(tok);
            if step % 20 == 0 || step + 1 == n_new {
                println!(
                    "[run {run}] step {step}: tok {tok} (logit {:.3}) s={} t={:.1}s",
                    logits[tok as usize],
                    ids.len(),
                    run_t0.elapsed().as_secs_f64()
                );
            }
        }
        let bin_path = out_dir.join(format!("gpu_logits_run{run}.bin"));
        std::fs::write(&bin_path, &logits_blob).expect("write logits bin");
        let sha = sha256_hex(&logits_blob);
        println!(
            "[run {run}] {} tokens in {:.1}s (informational, single-run, not a perf claim); logits bin {} sha256 {}",
            n_new,
            run_t0.elapsed().as_secs_f64(),
            bin_path.display(),
            sha
        );
        vram_line(&gpu, &format!("post-run{run}"));
        run_tokens.push(new_tokens);
        run_shas.push(sha);
    }

    let deterministic =
        run_tokens.windows(2).all(|w| w[0] == w[1]) && run_shas.windows(2).all(|w| w[0] == w[1]);
    println!(
        "determinism across {runs} runs: tokens {} | logits bins {}",
        if run_tokens.windows(2).all(|w| w[0] == w[1]) {
            "IDENTICAL"
        } else {
            "DIVERGENT"
        },
        if run_shas.windows(2).all(|w| w[0] == w[1]) {
            "BYTE-IDENTICAL"
        } else {
            "DIVERGENT"
        },
    );

    // bank the run record
    let mut f = std::fs::File::create(out_dir.join("gpu_greedy.json")).expect("json");
    let toks_json: Vec<String> = run_tokens
        .iter()
        .map(|r| {
            format!(
                "[{}]",
                r.iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            )
        })
        .collect();
    // flat per-run keys too (the hand-rolled JsonObj reader in memra-gguf consumes
    // flat u32 arrays; nested arrays are for humans)
    let flat_runs: String = run_tokens
        .iter()
        .enumerate()
        .map(|(i, r)| {
            format!(
                "  \"tokens_run{i}\": [{}],\n",
                r.iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            )
        })
        .collect();
    write!(
        f,
        "{{\n  \"variant\": \"{}\",\n  \"prompt\": [{}],\n  \"n_new\": {},\n  \"runs\": [{}],\n{}  \"logits_bin_sha256\": [{}],\n  \"deterministic\": {}\n}}\n",
        spec.variant_tag,
        prompt.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","),
        n_new,
        toks_json.join(", "),
        flat_runs,
        run_shas.iter().map(|s| format!("\"{s}\"")).collect::<Vec<_>>().join(", "),
        deterministic
    )
    .expect("write json");
    println!(
        "banked {} | total elapsed {:.0}s",
        out_dir.join("gpu_greedy.json").display(),
        t0.elapsed().as_secs_f64()
    );
    if !deterministic {
        println!("GPU GREEDY DRIVER: FAIL (nondeterministic across runs)");
        std::process::exit(1);
    }
    println!("GPU GREEDY DRIVER: complete (verdict comes from dsv4-greedy-verify)");
}
