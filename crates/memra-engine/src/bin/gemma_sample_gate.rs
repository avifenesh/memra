//! Gemma SAMPLED-ADMISSION gate (lane/gemma-sampled-spec, 2026-09-09): the flip receipt for
//! `MEMRA_GEMMA_SPEC_SAMPLED`. The T>0 rejection-sampling gemma assistant-drafter route must
//!
//!   (1) leave the greedy route untouched — a `Some(temp == 0)` config takes the greedy
//!       boundary and the greedy burst, byte for byte against `None` (kill-switch identity);
//!   (2) reproduce the greedy stream at T -> 0 (tiny-temperature continuity: the Gumbel noise
//!       is scaled by T, so every sampled arm — prime boundary, per-slot draft draw, accept
//!       walk, bonus draw, reject-slot residual — collapses onto the argmax it replaced);
//!   (3) emit tokens whose PER-POSITION distribution equals plain sampling from the same
//!       filtered target (two-sample chi-square across N independent seeds per arm);
//!   (4) REFUSE, loudly and by name, every shape the route does not implement: penalties on
//!       either arm, an FR-trimmed drafter head, and temp <= 0 into the sampled burst.
//!
//! This is the gemma twin of `dspark_sample_gate` and shares its statistical contract; what
//! differs is the route under test. gemma has no sampled one-shot `generate_spec_*`, so both
//! arms are driven through the SESSION api the worker actually serves on
//! (`gemma_spec_session_new` + `gemma_spec_session_burst{,_sampled}`) — the gate exercises the
//! served code path rather than a bin-only twin of it.
//!
//! Why a distribution gate and not a byte gate at (3): at T>0 the two arms consume different
//! Philox event sequences by construction, so only the DISTRIBUTION is the contract — which
//! is exactly the correctness claim of rejection sampling (Leviathan/Chen).
//!
//! Usage:
//!   gemma_sample_gate <target.gguf> <drafter.gguf> [t0|tiny|hist|refuse|all]
//! Env (hist mode):
//!   MEMRA_GSG_SEEDS   seeds per arm (default 600)
//!   MEMRA_GSG_TOKENS  positions compared (default 4)
//!   MEMRA_GSG_TEMP    temperature (default 1.0 — this family's VENDOR default)
//!   MEMRA_GSG_TOPK / MEMRA_GSG_TOPP  filters (default 64 / 0.95 — the vendor row)
//!   MEMRA_GSG_K       draft depth (default 5, the shipping K)
//!   MEMRA_GSG_PROMPT  comma-separated token ids (default: a fixed >= PRIME_MIN_T id prompt)
//!
//! RED ARM (how this gate was shown to fail before it was trusted): run it with
//! `MEMRA_GSG_RED=boundary` and the sampled prime draws its boundary token from the argmax
//! instead of the filtered row. (3) then breaks its bound at EVERY position while the unset
//! run passes — measured X2 380.7 / 246.0 / 142.5 / 163.9 against bounds 24.5 / 39.4 / 37.8 /
//! 43.9, so the check is non-vacuous.
//!
//! (2) is NOT the arm that catches this, and the distinction is worth stating because the
//! first draft of this file claimed it was: at T=1e-6 the sampled boundary collapses onto the
//! same argmax the greedy boundary takes, so tiny-T continuity is structurally blind to the
//! boundary seam and passes under the red arm. It gates the OTHER sampled arms (draft draw,
//! accept walk, bonus, residual). Only the distribution arm can see the boundary.
use memra_engine::Engine;
use memra_engine::gemma_spec::GemmaDraft;
use memra_engine::hybrid::HybridModel;
use memra_engine::spec::{SpecSampling, sample_boundary_token};
use memra_gguf::GgufFile;

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

fn sp_pen(temp: f32, seed: u64) -> SpecSampling {
    SpecSampling {
        temp,
        seed,
        top_k: 64,
        top_p: 0.95,
        min_p: 0.0,
        penalty_last_n: 8192,
        penalty_repeat: 1.1,
        penalty_freq: 0.5,
        penalty_present: 0.5,
    }
}

/// Drive the SESSION api the worker serves on, one burst of `max_new`. `sp = None` is the
/// greedy burst; `Some` is the sampled burst. The prime boundary is drawn under the same
/// config, exactly as `step_gemma_spec` does it.
fn session_stream(
    model: &HybridModel,
    e: &Engine,
    d: &mut GemmaDraft,
    prompt: &[u32],
    max_new: usize,
    k: usize,
    sp: Option<&SpecSampling>,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let eos: Vec<u32> = Vec::new();
    // RED ARM (MEMRA_GSG_RED=boundary): prime the session GREEDY while the burst still runs
    // sampled — i.e. put back the one-greedy-token-then-sampled shape the sampled boundary
    // was added to remove. Nothing else changes. This exists so the gate is known to be able
    // to fail: under it, (2) must diverge at position 0 and (3)'s pos 0 must break its bound.
    let prime_sp = if red_boundary() { None } else { sp };
    let mut sess =
        model.gemma_spec_session_new(e, d, prompt, prompt.len() + max_new + k + 8, prime_sp)?;
    let (out, _dr, _ac) = match sp {
        Some(cfg) => {
            model.gemma_spec_session_burst_sampled(e, d, &mut sess, max_new, k, &eos, cfg)?
        }
        None => model.gemma_spec_session_burst(e, d, &mut sess, max_new, k, &eos)?,
    };
    Ok(out)
}

/// Trunk-only sampled reference: plain serving-class decode, every token drawn from the
/// step's SOFTCAPPED logits row through the shipped filtered device sampler
/// (`sample_boundary_token` — the same filter_stats + gumbel_perturb_filtered composition the
/// spec accept walk targets). This IS "the trunk-only sampled distribution" at device truth,
/// and the gemma decode path softcaps its logits, so both arms see the same row.
fn plain_sampled(
    model: &HybridModel,
    e: &Engine,
    prompt: &[u32],
    max_new: usize,
    cfg: &SpecSampling,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let mut cache = memra_engine::pp::new_cache(e, &model.cfg, prompt.len() + max_new + 8)?;
    let mut logits = model.prime_cache(e, prompt, &mut cache, 0)?.0;
    let mut sctr = 0u32;
    let mut out = Vec::with_capacity(max_new);
    for _ in 0..max_new {
        let token = sample_boundary_token(e, &logits, cfg, &[], &mut sctr, "gemma trunk-ref")?;
        out.push(token);
        if out.len() >= max_new {
            break;
        }
        let mut caches = [&mut cache];
        logits = model.decode_step_batch(e, &[token], &mut caches)?.remove(0);
    }
    Ok(out)
}

/// The red arm switch. Read once; any value other than `boundary` (including unset) leaves
/// the gate in its real configuration.
fn red_boundary() -> bool {
    static R: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *R.get_or_init(|| std::env::var("MEMRA_GSG_RED").as_deref() == Ok("boundary"))
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
        .expect("usage: gemma_sample_gate <target.gguf> <drafter.gguf> [t0|tiny|hist|refuse|all]");
    let drafter = std::env::args().nth(2).expect("drafter gguf");
    let mode = std::env::args().nth(3).unwrap_or_else(|| "all".into());
    let e = Engine::new(0)?;
    let tg = GgufFile::open(&memra_gguf::hf::resolve_arg(&target)?)?;
    let model = HybridModel::load(&e, &tg)?;
    let dg = GgufFile::open(&memra_gguf::hf::resolve_arg(&drafter)?)?;
    let mut d = GemmaDraft::load(&e, &dg)?;
    let k = env_usize("MEMRA_GSG_K", 5);
    println!(
        "target {} layers, n_embd {}, vocab {}; drafter loaded; K={k}",
        model.cfg.n_layer,
        model.cfg.n_embd,
        model.output.out_features()
    );

    // The session prime takes prime_cache when the prompt clears PRIME_MIN_T, which is the
    // shape the worker serves; a shorter prompt would exercise the tokenwise fallback instead.
    let prompt: Vec<u32> = std::env::var("MEMRA_GSG_PROMPT")
        .ok()
        .map(|s| s.split(',').filter_map(|v| v.trim().parse().ok()).collect())
        .unwrap_or_else(|| {
            vec![
                2, 106, 1645, 108, 4385, 573, 2412, 3311, 235292, 235248, 235274, 235265, 235248,
                100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
            ]
        });
    assert!(
        prompt.len() >= memra_engine::hybrid_forward::PRIME_MIN_T,
        "MEMRA_GSG_PROMPT must clear PRIME_MIN_T ({}) so the gate primes the way the worker does",
        memra_engine::hybrid_forward::PRIME_MIN_T
    );
    if red_boundary() {
        println!(
            "RED ARM ACTIVE (MEMRA_GSG_RED=boundary): the sampled prime draws its boundary \
             token from the argmax. The HIST arm is expected to fail; a pass there means the \
             gate cannot see the seam it claims to gate. tiny-T is expected to still PASS — \
             at T=1e-6 both boundaries collapse onto the same argmax, so that arm is blind \
             to this seam by construction and gates the other sampled arms instead."
        );
    }
    let mut fails = 0usize;

    // --- (1) kill-switch identity: Some(temp=0) must take the greedy path byte for byte ---
    if mode == "t0" || mode == "all" {
        // What the kill-switch actually claims, stated so the check cannot be vacuous:
        // `temp == 0` belongs to the GREEDY burst on BOTH arms — the sampled burst refuses it
        // by name (covered in `refuse`), and `step_gemma_spec` never routes it there. So the
        // only place a temp-0 config can still perturb the greedy stream is the PRIME
        // BOUNDARY, whose `match sp { Some(sp) if sp.temp > 0.0 => sample, _ => argmax }` is
        // the one live branch. That is what is compared here. Comparing the greedy stream to
        // another greedy stream would prove nothing about the seam, so the second greedy run
        // is reported for what it is: a determinism control on the route itself.
        let cfg0 = sp(0.0, 42, 0, 1.0);
        let s_none = model.gemma_spec_session_new(&e, &mut d, &prompt, prompt.len() + 80, None)?;
        let s_zero =
            model.gemma_spec_session_new(&e, &mut d, &prompt, prompt.len() + 80, Some(&cfg0))?;
        let boundary_same = s_none.pending_token() == s_zero.pending_token();
        let a = session_stream(&model, &e, &mut d, &prompt, 64, k, None)?;
        let b = session_stream(&model, &e, &mut d, &prompt, 64, k, None)?;
        let deterministic = a == b && !a.is_empty();
        let ok = boundary_same && deterministic;
        println!(
            "t0 kill-switch: prime boundary None vs Some(temp=0) {} (pending {} vs {}); greedy \
             determinism control {}: {}",
            if boundary_same {
                "IDENTICAL"
            } else {
                "DIVERGED"
            },
            s_none.pending_token(),
            s_zero.pending_token(),
            if deterministic { "EXACT" } else { "DIVERGED" },
            if ok {
                "OK"
            } else {
                fails += 1;
                "FAIL"
            }
        );
    }

    // --- (4) refusals: shapes the route does not implement must fail loudly, by name ---
    if mode == "refuse" || mode == "all" {
        let cfg = sp(1.0, 42, 64, 0.95);
        let mut sess =
            model.gemma_spec_session_new(&e, &mut d, &prompt, prompt.len() + 80, Some(&cfg))?;
        let eos: Vec<u32> = Vec::new();

        let zero = sp(0.0, 42, 0, 1.0);
        let temp0_refused = matches!(
            model.gemma_spec_session_burst_sampled(&e, &mut d, &mut sess, 8, k, &eos, &zero),
            Err(err) if err.to_string().contains("temp <= 0 is the greedy burst's shape")
        );
        let pen = sp_pen(1.0, 42);
        let pen_refused = matches!(
            model.gemma_spec_session_burst_sampled(&e, &mut d, &mut sess, 8, k, &eos, &pen),
            Err(err) if err.to_string().contains("penalties are not admitted on this route")
        );
        // An FR-trimmed drafter head maps draft rows to target ids, so the recorded q would be
        // a distribution over a different support than the verify row it is tested against.
        let saved = d.d2t.take();
        d.d2t = Some(vec![0u32; 8]);
        let trim_refused = matches!(
            model.gemma_spec_session_burst_sampled(&e, &mut d, &mut sess, 8, k, &eos, &cfg),
            Err(err) if err.to_string().contains("FR-trimmed drafter head")
        );
        d.d2t = saved;

        let ok = temp0_refused && pen_refused && trim_refused;
        println!(
            "refusals (temp<=0: {temp0_refused}, penalties: {pen_refused}, d2t trim: \
             {trim_refused}): {}",
            if ok {
                "OK"
            } else {
                fails += 1;
                "FAIL (an unimplemented shape must refuse by name, never serve silently)"
            }
        );
    }

    // --- (2) tiny-temperature continuity: T=1e-6 must byte-reproduce the greedy stream ---
    if mode == "tiny" || mode == "all" {
        let a = session_stream(&model, &e, &mut d, &prompt, 64, k, None)?;
        for seed in [42u64, 7, 1234] {
            let cfgt = sp(1e-6, seed, 0, 1.0);
            let b = session_stream(&model, &e, &mut d, &prompt, 64, k, Some(&cfgt))?;
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

    // --- (3) per-position distribution: spec-on vs trunk-only, N independent seeds per arm ---
    if mode == "hist" || mode == "all" {
        let n_seeds = env_usize("MEMRA_GSG_SEEDS", 600);
        let m_tok = env_usize("MEMRA_GSG_TOKENS", 4);
        let temp = env_f32("MEMRA_GSG_TEMP", 1.0);
        let top_k = env_usize("MEMRA_GSG_TOPK", 64) as i32;
        let top_p = env_f32("MEMRA_GSG_TOPP", 0.95);
        println!(
            "hist: {n_seeds} seeds/arm, {m_tok} positions, temp {temp}, top_k {top_k}, \
             top_p {top_p}, K {k}, prompt len {}",
            prompt.len()
        );
        use std::collections::HashMap;
        let mut hist_a: Vec<HashMap<u32, u64>> = vec![HashMap::new(); m_tok];
        let mut hist_b: Vec<HashMap<u32, u64>> = vec![HashMap::new(); m_tok];
        let t0 = std::time::Instant::now();
        for i in 0..n_seeds {
            // disjoint seed spaces: the arms must be INDEPENDENT samples of the target
            let cfg_a = sp(temp, 1_000_000 + i as u64, top_k, top_p);
            let a = plain_sampled(&model, &e, &prompt, m_tok, &cfg_a)?;
            for (j, &t) in a.iter().take(m_tok).enumerate() {
                *hist_a[j].entry(t).or_insert(0) += 1;
            }
            let cfg_b = sp(temp, 9_000_000 + i as u64, top_k, top_p);
            let b = session_stream(&model, &e, &mut d, &prompt, m_tok, k, Some(&cfg_b))?;
            for (j, &t) in b.iter().take(m_tok).enumerate() {
                *hist_b[j].entry(t).or_insert(0) += 1;
            }
            if (i + 1) % 200 == 0 {
                println!(
                    "  ... {} / {n_seeds} seeds ({:.0}s)",
                    i + 1,
                    t0.elapsed().as_secs_f64()
                );
            }
        }
        // Two-sample chi-square per position: bucket = tokens with combined count >= 10 (rare
        // tail pooled), X^2 = sum (a-b)^2/(a+b) ~ chi2(df=buckets-1) under the null (equal N).
        // Bound = 0.999 quantile — stated power, printed per position.
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
                "pos {j}: X2={x2:.1} df={df:.0} bound(q=.999)={bound:.1} TV={tvn:.4} support={} {}",
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
        "== gemma_sample_gate: {} ==",
        if fails == 0 { "ALL PASS" } else { "FAIL" }
    );
    std::process::exit(if fails == 0 { 0 } else { 1 });
}
