//! DSpark SAMPLED-ADMISSION gate (lane/dspark-sampled-admission-20260820): the T>0
//! rejection-sampling dspark route must (1) keep T=0 byte-identical to the greedy route
//! (kill-switch identity), (2) reproduce the greedy stream at T->0 (tiny-temperature
//! continuity — the frspec gate-(1) shape: Gumbel noise scaled by T vanishes), and
//! (3) emit tokens whose PER-POSITION distribution equals trunk-only sampling from the
//! same filtered target (two-sample chi-square across N seeds, spec-on vs a plain-decode
//! reference that draws through the same shipped device sampler primitives).
//!
//! Usage:
//!   dspark_sample_gate <target.gguf|hf_dir> <draft_export_dir> [t0|tiny|hist|all]
//! Env (hist mode):
//!   MEMRA_SG_SEEDS   seeds per arm (default 2000)
//!   MEMRA_SG_TOKENS  positions compared (default 4)
//!   MEMRA_SG_TEMP    temperature (default 0.8)
//!   MEMRA_SG_TOPK / MEMRA_SG_TOPP  filters (default 0 / 1.0 — pure temp; set 20/0.95
//!                    for the vendor-keys regime)
//!   MEMRA_SG_REP / MEMRA_SG_FREQ / MEMRA_SG_PRESENT  penalties (default 1.0/0.0/0.0 —
//!                    off; lane/dspark-penalized-sampled-20260821: both arms carry them,
//!                    the plain arm through the same boundary-sampler penalty pass, so
//!                    hist gates the PENALIZED spec route against penalized trunk-only)
//!   MEMRA_SG_LASTN   penalty window (default: 8192, the serve API's cross-path bound;
//!                    ignored when penalties are off)
//!   MEMRA_SG_PROMPT  comma-separated token ids (default a fixed short id prompt)
//!
//! t0 mode also gates the PENALTIES-AT-T=0 defined behavior: a temp==0 config carrying
//! non-identity penalties must be REFUSED by both the bin arm and the serve session
//! (the greedy walk argmaxes raw columns; penalized greedy is served on the plain path).
//!
//! Why a distribution gate and not a byte gate: at T>0 the two arms consume different
//! Philox event sequences by construction, so only the DISTRIBUTION is the contract —
//! per-position token histograms over many independent seeds, exactly the correctness
//! claim of rejection sampling (Leviathan/Chen; the composition arm of sample_check pins
//! the same claim at kernel scale, the CPU tests in dflash.rs pin the walk math).
use memra_engine::Engine;
use memra_engine::dflash::DflashDraft;
use memra_engine::hybrid::HybridModel;
use memra_engine::spec::{SpecSampling, sample_boundary_token};
use memra_gguf::GgufFile;

const DEFAULT_PENALTY_WINDOW: usize = memra_engine::spec::PEN_WINDOW_MAX;

fn load_model(path: &str, e: &Engine) -> Result<HybridModel, Box<dyn std::error::Error>> {
    let path = memra_gguf::hf::resolve_arg(path)?;
    let is_dir = std::path::Path::new(&path).is_dir();
    if is_dir {
        let dir = std::path::Path::new(&path);
        if dir.join("manifest.json").exists() {
            let repack = memra_gguf::source::Hy3RepackSource::open(dir)?;
            Ok(HybridModel::load_from_source(e, &repack)?)
        } else {
            let src = memra_gguf::source::SafetensorsSource::open(dir)?;
            Ok(HybridModel::load_from_source(e, &src)?)
        }
    } else {
        let g = GgufFile::open(&path)?;
        Ok(HybridModel::load(e, &g)?)
    }
}

fn sp(temp: f32, seed: u64, top_k: i32, top_p: f32) -> SpecSampling {
    SpecSampling {
        temp,
        seed,
        top_k,
        top_p,
        min_p: 0.0,
        penalty_last_n: 0,
        penalty_repeat: 1.0,
        penalty_freq: 0.0,
        penalty_present: 0.0,
    }
}

/// Penalized variant: `last_n` defaults to the serve API's arming (whole context) when
/// any coefficient is non-identity.
#[allow(clippy::too_many_arguments)]
fn sp_pen(
    temp: f32,
    seed: u64,
    top_k: i32,
    top_p: f32,
    rep: f32,
    freq: f32,
    present: f32,
    last_n: usize,
) -> SpecSampling {
    let pen_on = rep != 1.0 || freq != 0.0 || present != 0.0;
    SpecSampling {
        temp,
        seed,
        top_k,
        top_p,
        min_p: 0.0,
        penalty_last_n: if pen_on { last_n } else { 0 },
        penalty_repeat: rep,
        penalty_freq: freq,
        penalty_present: present,
    }
}

/// Trunk-only sampled reference: plain serving-class decode, every token drawn from the
/// step's logits row through the SHIPPED filtered device sampler (`sample_boundary_token`
/// — the same filter_stats + gumbel_perturb_filtered composition the spec accept walk
/// targets). This IS "the trunk-only sampled distribution" at device-truth semantics.
fn plain_sampled(
    model: &HybridModel,
    e: &Engine,
    prompt: &[u32],
    max_new: usize,
    cfg: &SpecSampling,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let mut cache = memra_engine::pp::new_cache(e, &model.cfg, prompt.len() + max_new + 8)?;
    let mut logits = if prompt.len() >= memra_engine::hybrid_forward::PRIME_MIN_T {
        model.prime_cache(e, prompt, &mut cache, 0)?.0
    } else {
        let mut row = Vec::new();
        for &token in prompt {
            let mut caches = [&mut cache];
            row = model.decode_step_batch(e, &[token], &mut caches)?.remove(0);
        }
        row
    };
    let mut sctr = 0u32;
    let mut out = Vec::with_capacity(max_new);
    // Penalized window (lane/dspark-penalized-sampled-20260821): the same seed + trim the
    // route ships (pen_window_seed; sample_boundary_token trims to min(last_n, cap) at
    // every draw) — this arm IS penalized trunk-only sampling at device-truth semantics.
    let mut pen_hist: Vec<u32> = if cfg.pen_on() {
        memra_engine::spec::pen_window_seed(&[], prompt, cfg.penalty_last_n)
    } else {
        Vec::new()
    };
    for _ in 0..max_new {
        let token = sample_boundary_token(e, &logits, cfg, &pen_hist, &mut sctr, "trunk-ref")?;
        out.push(token);
        if cfg.pen_on() {
            pen_hist.push(token);
        }
        if out.len() >= max_new {
            break;
        }
        let mut caches = [&mut cache];
        logits = model.decode_step_batch(e, &[token], &mut caches)?.remove(0);
    }
    Ok(out)
}

fn first_divergence(a: &[u32], b: &[u32]) -> Option<usize> {
    let n = a.len().min(b.len());
    (0..n)
        .find(|&i| a[i] != b[i])
        .or(if a.len() != b.len() { Some(n) } else { None })
}

/// chi-square 0.999 quantile (Wilson–Hilferty approximation) — the per-position bound.
fn chi2_q999(df: f64) -> f64 {
    let z = 3.0902; // Phi^-1(0.999)
    df * (1.0 - 2.0 / (9.0 * df) + z * (2.0 / (9.0 * df)).sqrt()).powi(3)
}

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}
fn env_f32(k: &str, d: f32) -> f32 {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let target = std::env::args()
        .nth(1)
        .expect("usage: dspark_sample_gate <target> <draft_dir> [t0|tiny|hist|all]");
    let draft_dir = std::env::args().nth(2).expect("draft export dir");
    let mode = std::env::args().nth(3).unwrap_or_else(|| "all".into());
    let e = Engine::new(0)?;
    let model = load_model(&target, &e)?;
    let draft = DflashDraft::load(&e, std::path::Path::new(&draft_dir))?;
    println!(
        "target loaded ({} layers); draft block {}, dflash2 {}, markov {}",
        model.cfg.n_layer,
        draft.cfg.block_size,
        draft.dflash2.is_some(),
        draft.markov.is_some()
    );
    // Default prompt: 24 fixed ids — generate_spec_dspark primes through prime_cache,
    // which requires T >= 16 (hybrid_forward's caller gate), so the id prompt must
    // clear that floor (the dspark_q38_gate 6-id prompts ride the same floor only via
    // MEMRA_PROMPT_DIR's longer chat renders).
    let prompt: Vec<u32> = std::env::var("MEMRA_SG_PROMPT")
        .ok()
        .map(|s| s.split(',').filter_map(|v| v.trim().parse().ok()).collect())
        .unwrap_or_else(|| {
            vec![
                84270, 279, 2701, 7355, 25, 220, 16, 13, 220, 100, 101, 102, 103, 104, 105, 106,
                107, 108, 109, 110, 111, 112, 113, 114,
            ]
        });
    assert!(
        prompt.len() >= 16,
        "MEMRA_SG_PROMPT must carry >= 16 token ids (prime_cache floor)"
    );
    let eos: Vec<u32> = Vec::new();
    let mut fails = 0usize;

    // --- kill-switch identity: Some(temp=0) must take the greedy code path byte-for-byte ---
    if mode == "t0" || mode == "all" {
        let a = model.generate_spec_dspark(&e, &draft, &prompt, 64, &eos, None)?;
        let cfg0 = sp(0.0, 42, 0, 1.0);
        let b = model.generate_spec_dspark(&e, &draft, &prompt, 64, &eos, Some(&cfg0))?;
        let ok = a == b && !a.is_empty();
        println!(
            "t0 kill-switch (None == Some(temp=0)): {}",
            if ok {
                "EXACT"
            } else {
                fails += 1;
                "DIVERGED"
            }
        );
        // Penalties at T=0: DEFINED behavior = a LOUD refusal on both entry points (the
        // greedy walk argmaxes RAW columns and would silently drop the penalties; the
        // worker serves penalized greedy on the plain path). A silent greedy stream here
        // is the H-class program switch this gate exists to catch.
        let cfg0p = sp_pen(0.0, 42, 0, 1.0, 1.1, 0.5, 0.5, usize::MAX);
        let bin_refused = matches!(
            model.generate_spec_dspark(&e, &draft, &prompt, 8, &eos, Some(&cfg0p)),
            Err(err) if err.to_string().contains("penalized greedy is served on the plain path")
        );
        let sess_refused = matches!(
            model.dspark_spec_session_new(
                &e,
                &draft,
                &prompt,
                prompt.len() + 128,
                Some(cfg0p),
                false,
            ),
            Err(err) if err.to_string().contains("penalized greedy is served on the plain path")
        );
        let ok = bin_refused && sess_refused;
        println!(
            "t0+penalties defined behavior (bin refuses: {bin_refused}, session refuses: \
             {sess_refused}): {}",
            if ok {
                "OK"
            } else {
                fails += 1;
                "FAIL (temp==0 + penalties must refuse by name)"
            }
        );
    }

    // --- tiny-temperature continuity: the SAMPLED code path at T=1e-6 must byte-reproduce
    // the greedy stream (all sampled arms exercised: boundary, chain/selector, accept walk,
    // bonus/residual — with distributions collapsed onto the argmax). ---
    if mode == "tiny" || mode == "all" {
        for seed in [42u64, 7, 1234] {
            let a = model.generate_spec_dspark(&e, &draft, &prompt, 64, &eos, None)?;
            let cfgt = sp(1e-6, seed, 0, 1.0);
            let b = model.generate_spec_dspark(&e, &draft, &prompt, 64, &eos, Some(&cfgt))?;
            let div = first_divergence(&a, &b);
            let ok = div.is_none() && !a.is_empty();
            println!(
                "tiny-T continuity seed {seed}: {}",
                if ok {
                    "EXACT".into()
                } else {
                    fails += 1;
                    format!("DIVERGED at {div:?}")
                }
            );
        }
    }

    // --- per-position distribution: spec-on vs trunk-only, N independent seeds per arm ---
    if mode == "hist" || mode == "all" {
        let n_seeds = env_usize("MEMRA_SG_SEEDS", 2000);
        let m_tok = env_usize("MEMRA_SG_TOKENS", 4);
        let temp = env_f32("MEMRA_SG_TEMP", 0.8);
        let top_k = env_usize("MEMRA_SG_TOPK", 0) as i32;
        let top_p = env_f32("MEMRA_SG_TOPP", 1.0);
        let rep = env_f32("MEMRA_SG_REP", 1.0);
        let freq = env_f32("MEMRA_SG_FREQ", 0.0);
        let present = env_f32("MEMRA_SG_PRESENT", 0.0);
        let last_n = env_usize("MEMRA_SG_LASTN", DEFAULT_PENALTY_WINDOW);
        println!(
            "hist: {n_seeds} seeds/arm, {m_tok} positions, temp {temp}, top_k {top_k}, \
             top_p {top_p}, rep {rep}, freq {freq}, present {present}, last_n {last_n}, \
             prompt len {}",
            prompt.len()
        );
        use std::collections::HashMap;
        let mut hist_a: Vec<HashMap<u32, u64>> = vec![HashMap::new(); m_tok];
        let mut hist_b: Vec<HashMap<u32, u64>> = vec![HashMap::new(); m_tok];
        let t0 = std::time::Instant::now();
        for i in 0..n_seeds {
            // disjoint seed spaces: the arms must be INDEPENDENT samples of the target
            let cfg_a = sp_pen(
                temp,
                1_000_000 + i as u64,
                top_k,
                top_p,
                rep,
                freq,
                present,
                last_n,
            );
            let a = plain_sampled(&model, &e, &prompt, m_tok, &cfg_a)?;
            for (j, &t) in a.iter().take(m_tok).enumerate() {
                *hist_a[j].entry(t).or_insert(0) += 1;
            }
            let cfg_b = sp_pen(
                temp,
                9_000_000 + i as u64,
                top_k,
                top_p,
                rep,
                freq,
                present,
                last_n,
            );
            let b = model.generate_spec_dspark(&e, &draft, &prompt, m_tok, &eos, Some(&cfg_b))?;
            for (j, &t) in b.iter().take(m_tok).enumerate() {
                *hist_b[j].entry(t).or_insert(0) += 1;
            }
            if (i + 1) % 500 == 0 {
                println!(
                    "  ... {} / {n_seeds} seeds ({:.0}s)",
                    i + 1,
                    t0.elapsed().as_secs_f64()
                );
            }
        }
        // Two-sample chi-square per position: bucket = tokens with combined count >= 10
        // (rare tail pooled), X^2 = sum (a-b)^2/(a+b) ~ chi2(df=buckets-1) under the null
        // (equal N). Bound = 0.999 quantile — stated power, printed per position.
        for j in 0..m_tok {
            let mut tokens: std::collections::HashSet<u32> = hist_a[j].keys().copied().collect();
            tokens.extend(hist_b[j].keys().copied());
            let (mut x2, mut buckets) = (0f64, 0usize);
            let (mut tail_a, mut tail_b) = (0f64, 0f64);
            let mut tv = 0f64;
            for &t in &tokens {
                let a = *hist_a[j].get(&t).unwrap_or(&0) as f64;
                let b = *hist_b[j].get(&t).unwrap_or(&0) as f64;
                tv += (a - b).abs();
                if a + b >= 10.0 {
                    x2 += (a - b) * (a - b) / (a + b);
                    buckets += 1;
                } else {
                    tail_a += a;
                    tail_b += b;
                }
            }
            if tail_a + tail_b > 0.0 {
                x2 += (tail_a - tail_b) * (tail_a - tail_b) / (tail_a + tail_b);
                buckets += 1;
            }
            let df = (buckets.max(2) - 1) as f64;
            let bound = chi2_q999(df);
            let tvn = tv / (2.0 * n_seeds as f64);
            let ok = x2 < bound;
            println!(
                "pos {j}: X2={x2:.1} df={df:.0} bound(q=.999)={bound:.1} TV={tvn:.4} \
                 support={} {}",
                tokens.len(),
                if ok {
                    "OK"
                } else {
                    fails += 1;
                    "FAIL"
                }
            );
        }
    }

    println!(
        "== dspark_sample_gate: {} ==",
        if fails == 0 { "ALL PASS" } else { "FAIL" }
    );
    std::process::exit(if fails == 0 { 0 } else { 1 });
}

#[cfg(test)]
mod tests {
    #[test]
    fn default_penalty_window_matches_the_served_cross_path_bound() {
        assert_eq!(super::DEFAULT_PENALTY_WINDOW, 8192);
    }
}
