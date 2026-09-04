//! argmax-margin-probe (lane/q8-argmax, 2026-08-06): calibrates the run-gen prefill-vs-decode
//! argmax gate against the ONE number that decides whether a flip is a near-tie coin or a real
//! numeric defect — the top-2 MARGIN at the decision position, measured against the margin
//! distribution over that same prompt's other positions.
//!
//! Why this exists: the gate (`run_gen.rs:880`) prints `logit maxdiff`, and a reader naturally
//! treats a big maxdiff as "big error". That is wrong, and it cost this lane's predecessors real
//! time. maxdiff is the max over a 248k-wide vocab of |prefill - decode| — dominated by
//! whatever logit is noisiest anywhere in the vocab, mostly deep in the tail, and it is
//! routinely 0.3-2.4 on runs the same gate calls MATCH. The gate flips iff the top-2 margin at
//! the last position is SMALLER than the config spread there. So the discriminator is
//!     margin(top1, top2)  vs  |prefill[i] - decode[i]| at the two contending ids
//! and the honest way to state "this is a near-tie" is to show that margin sits in the extreme
//! low tail of the margins this same prompt produces at every other position.
//!
//! It reports, for the batched-prefill config and the tokenwise-decode config:
//!   * per-position top-2 margin over the last WINDOW positions of the prompt (teacher-forced
//!     on the prompt itself — bit-identical inputs, no stream separation),
//!   * the decision position's margin and its PERCENTILE within that distribution,
//!   * the per-id config delta at the two contending ids (the quantity that must exceed the
//!     margin for a flip to be possible at all),
//!   * agreement rate across all sampled positions (a near-tie class predicts ~1 flip in many;
//!     a broken kernel predicts systematic disagreement).
//!
//! usage: argmax-margin-probe <model.gguf|hf_dir> <prompt-file> [window=24]
//! env: engine knobs (MEMRA_FA_SPLIT, MEMRA_Q80_G2, MEMRA_FAST, ...) steer the arm under test.

use memra_engine::Engine;
use memra_engine::cache::Cache;
use memra_engine::hybrid::HybridModel;
use memra_gguf::GgufFile;
use memra_tokenizer::Tokenizer;

fn top2(l: &[f32]) -> (usize, f32, usize, f32) {
    let (mut i1, mut v1, mut i2, mut v2) = (0usize, f32::NEG_INFINITY, 0usize, f32::NEG_INFINITY);
    for (i, &v) in l.iter().enumerate() {
        if v > v1 {
            i2 = i1;
            v2 = v1;
            i1 = i;
            v1 = v;
        } else if v > v2 {
            i2 = i;
            v2 = v;
        }
    }
    (i1, v1, i2, v2)
}

fn measurement_row(pos: usize, prefill: (usize, f32), decode: (usize, f32), delta: f32) -> String {
    let (p1, mp) = prefill;
    let (d1, md) = decode;
    // The shell gate compares these values, so display rounding must not erase a
    // small explained margin. Ten significant digits preserve every finite f32.
    format!(
        "{pos:<8} {p1:<13} {mp:<16.9e} {d1:<12} {md:<16.9e} {delta:<16.9e} {}",
        if p1 == d1 { "yes" } else { "NO <-- FLIP" }
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model_path = args
        .first()
        .expect("usage: argmax-margin-probe <model.gguf|hf_dir> <prompt-file> [window]");
    let prompt_file = args.get(1).expect("need <prompt-file>");
    let window: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(24);
    if window == 0 {
        return Err("window must be a positive integer".into());
    }

    let e = Engine::new(0)?;
    let path = std::path::Path::new(model_path);
    let (model, tok) = if path.is_dir() {
        let source = memra_gguf::source::SafetensorsSource::open(path)?;
        (
            HybridModel::load_from_source_without_mtp(&e, &source)?,
            Tokenizer::from_hf_dir(path).map_err(|err| format!("tokenizer: {err}"))?,
        )
    } else {
        let g = GgufFile::open(model_path)?;
        (
            HybridModel::load_without_mtp(&e, &g)?,
            Tokenizer::from_gguf(&g).map_err(|err| format!("tokenizer: {err}"))?,
        )
    };
    let text = std::fs::read_to_string(prompt_file)?;
    let prompt = tok.encode(&text, true);
    let t = prompt.len();
    assert!(
        window < t.saturating_sub(2),
        "prompt shorter than the window"
    );

    println!(
        "argmax-margin-probe: T={t} window={window} sm_count={} model={}",
        e.sm_count(),
        model_path
    );
    println!(
        "env: MEMRA_FA_SPLIT={} MEMRA_Q80_G2={} MEMRA_FAST={} MEMRA_PRIME_CHUNK={}",
        std::env::var("MEMRA_FA_SPLIT").unwrap_or_else(|_| "<unset>".into()),
        std::env::var("MEMRA_Q80_G2").unwrap_or_else(|_| "<unset>".into()),
        std::env::var("MEMRA_FAST").unwrap_or_else(|_| "<unset>".into()),
        std::env::var("MEMRA_PRIME_CHUNK").unwrap_or_else(|_| "<unset>".into()),
    );

    // --- config A: the tokenwise decode path, capturing logits at EVERY position ---
    // This is the gate's "decode" side. Stepping the whole prompt gives us, for free, the
    // per-position top-2 margin distribution under one fixed arithmetic config.
    let mut cache = Cache::new(&e, &model.cfg, t + 8)?;
    let mut dec_at: Vec<Vec<f32>> = Vec::with_capacity(window);
    for (i, &tk) in prompt.iter().enumerate() {
        let l = model.decode_step(&e, tk, &mut cache)?;
        if i + window >= t {
            dec_at.push(l);
        }
    }

    // --- config B: the batched prefill path (the gate's "prefill" side), one forward per
    //     truncation length so we get the SAME positions under the other config. This is the
    //     expensive half (window forwards over a ~2k prompt) — the honest way, since
    //     forward_last only returns the last row.
    let mut pre_at: Vec<Vec<f32>> = Vec::with_capacity(window);
    for i in (t - window)..t {
        pre_at.push(model.forward_last(&e, &prompt[..=i])?);
    }

    // --- per-position comparison ---
    println!(
        "\npos      prefill_top1  margin_p     decode_top1  margin_d     delta@ids      agree"
    );
    let mut margins_d: Vec<f32> = Vec::with_capacity(window);
    let mut margins_p: Vec<f32> = Vec::with_capacity(window);
    let mut n_agree = 0usize;
    let mut flip_rows: Vec<(usize, f32, f32, bool)> = Vec::new();
    let mut last_row: Option<(usize, usize, f32, f32, f32, f32)> = None;
    for k in 0..window {
        let pos = t - window + k;
        let (p1, pv1, p2, pv2) = top2(&pre_at[k]);
        let (d1, dv1, d2, dv2) = top2(&dec_at[k]);
        let mp = pv1 - pv2;
        let md = dv1 - dv2;
        margins_p.push(mp);
        margins_d.push(md);
        // the config delta at the contending ids — what must exceed the margin to flip
        let ids = [p1, p2, d1, d2];
        let delta = ids
            .iter()
            .map(|&i| (pre_at[k][i] - dec_at[k][i]).abs())
            .fold(0.0f32, f32::max);
        let agree = p1 == d1;
        if agree {
            n_agree += 1;
        } else {
            // per-flip classification (gemma-31B calibration, 2026-08-17): a flip is
            // margin-EXPLAINED iff the config spread at the contending ids can reach
            // across the smaller of the two margins at ITS OWN position.
            flip_rows.push((pos, mp.min(md), delta, delta > mp.min(md)));
        }
        println!("{}", measurement_row(pos, (p1, mp), (d1, md), delta));
        if k == window - 1 {
            last_row = Some((p1, d1, mp, md, delta, (pre_at[k][p1] - dec_at[k][p1]).abs()));
        }
    }

    // --- the calibration verdict: where does the decision position's margin sit? ---
    let mut sorted = margins_d.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let pct = |q: f64| sorted[((sorted.len() - 1) as f64 * q).round() as usize];
    let (fp1, fd1, fmp, fmd, fdelta, _) = last_row.expect("window >= 1");
    let below = margins_d.iter().filter(|&&m| m < fmd).count();
    println!(
        "\nmargin distribution (decode config, {} positions): min {:.4}  p10 {:.4}  p50 {:.4}  p90 {:.4}  max {:.4}",
        sorted.len(),
        sorted[0],
        pct(0.10),
        pct(0.50),
        pct(0.90),
        sorted[sorted.len() - 1]
    );
    let mut sp = margins_p.clone();
    sp.sort_by(|a, b| a.total_cmp(b));
    println!(
        "margin distribution (prefill config): min {:.4}  p50 {:.4}  max {:.4}",
        sp[0],
        sp[sp.len() / 2],
        sp[sp.len() - 1]
    );
    println!(
        "DECISION POSITION (the gate's): prefill_top1={fp1} decode_top1={fd1} margin_p={fmp:.4} \
         margin_d={fmd:.4} config_delta_at_ids={fdelta:.4}"
    );
    println!(
        "  -> the decision margin is BELOW {}/{} sampled positions' margins (rank {} of {})",
        below,
        margins_d.len(),
        below + 1,
        margins_d.len()
    );
    println!(
        "  -> flip is ARITHMETICALLY POSSIBLE iff config_delta > margin: {:.4} > {:.4} = {}",
        fdelta,
        fmd.min(fmp),
        fdelta > fmd.min(fmp)
    );
    println!(
        "agreement across sampled positions: {}/{} ({} flip(s))",
        n_agree,
        window,
        window - n_agree
    );
    // Classification. The two independent axes are (a) how many positions flipped, and
    // (b) whether the config spread at the contending ids is even large enough to reach
    // across the decision margin. Only (flip AND spread < margin) is a defect.
    let flips = window - n_agree;
    let exposed = fdelta > fmd.min(fmp);
    // PER-MODEL FLIP BUDGET (calibration rows). The old verdict hard-coded flips>=2 =>
    // SYSTEMATIC regardless of margins — a stale calibration for models with a wide
    // near-tie population. gemma4-31B row (2026-08-17, calibrated from the banked
    // board-2048 margin measurement, gemma-ship-20260817/zoofusion/gate-*.log: decode
    // margins p10 0.293 / p50 2.28 with config spreads at contending ids up to 3.75 —
    // ~5 of 12 tail positions sit with margin inside the spread, a near-tie coin
    // population whose expectation is ~2.5 flips per 12-window; budget = 3 EXPLAINED
    // flips per 12 sampled positions, scaled by window). Every flip must STILL be
    // individually margin-explained — one unexplained flip is a defect at any count;
    // the budget only governs how many explained near-tie coins may land differently.
    let flip_budget = {
        let per12: usize = if model.cfg.gemma4.is_some() && model.cfg.n_embd == 5376 {
            3 // gemma4-31B calibration row (measured near-tie population, NOT loosened-to-green)
        } else {
            1 // pre-existing default budget for the thin-near-tie models (qwen q8 class)
        };
        (per12 * window).div_ceil(12)
    };
    let unexplained: Vec<_> = flip_rows.iter().filter(|r| !r.3).collect();
    for (pos, margin, delta, expl) in &flip_rows {
        println!(
            "flip@{pos}: margin {margin:.4} vs config spread {delta:.4} -> {}",
            if *expl {
                "margin-EXPLAINED (near-tie coin)"
            } else {
                "UNEXPLAINED"
            }
        );
    }
    println!(
        "flip budget: {flips} flip(s) vs budget {flip_budget} (calibration row: {})",
        if model.cfg.gemma4.is_some() && model.cfg.n_embd == 5376 {
            "gemma4-31B (3/12-window)"
        } else {
            "default (1/12-window)"
        }
    );
    println!(
        "VERDICT-INPUT: {}",
        if !unexplained.is_empty() {
            "UNEXPLAINED — flip(s) at margins the config spread does NOT cover; a real defect, investigate"
        } else if flips > flip_budget {
            "SYSTEMATIC — explained flips exceed the model's calibrated near-tie budget; investigate"
        } else if flips > 0 {
            "NEAR-TIE class — every flip sits at a margin the config spread covers, within the \
             model's calibrated budget (documented cross-config drift, not a numeric defect)"
        } else if exposed {
            "NEAR-TIE-EXPOSED — no flip fired, but the margin is inside the config spread: \
             this position is a coin that happened to land the same way in both configs"
        } else {
            "STABLE — no flip, and the config spread cannot reach across the decision \
             margin (structurally safe at this position)"
        }
    );
    // REPORTER ONLY: the gate (tools/argmax-margin-gate.sh) parses the table above and
    // owns the exit code — a probe-side exit would double-fail ahead of the comparator
    // and break the canary's inject-then-parse flow.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::measurement_row;

    #[test]
    fn table_round_trips_f32_margins() {
        for (margin, delta) in [
            (0.00002_f32, 0.00003_f32),
            (1.0e-30, 2.0e-30),
            (1.2345678, 2.3456788),
        ] {
            let row = measurement_row(10, (1, margin), (2, margin), delta);
            let fields: Vec<_> = row.split_whitespace().collect();
            assert_eq!(
                fields[2].parse::<f32>().unwrap().to_bits(),
                margin.to_bits()
            );
            assert_eq!(
                fields[4].parse::<f32>().unwrap().to_bits(),
                margin.to_bits()
            );
            assert_eq!(fields[5].parse::<f32>().unwrap().to_bits(), delta.to_bits());
            assert_eq!(fields[6], "NO");
            assert!(fields[5].parse::<f64>().unwrap() > fields[2].parse::<f64>().unwrap());
        }
        assert_eq!(
            measurement_row(10, (1, 1.0), (1, 1.0), 0.0)
                .split_whitespace()
                .nth(6),
            Some("yes")
        );
    }
}
