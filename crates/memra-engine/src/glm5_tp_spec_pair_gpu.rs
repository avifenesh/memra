//! Real-artifact pair oracles, deliberately ignored on ordinary test runs.
//! Run each arm in a fresh process through the lane's run-pair-cell.sh. A same-device
//! fixture cannot exercise the symmetric peer-AR program. Required inputs are
//! TP_SPEC_MODEL, TP_SPEC_PROMPT_IDS, TP_SPEC_OUT and TP_SPEC_ARM=plain|tp|pp.
//! O1 here checks the engine plain route with the spec door off/on. The companion
//! HTTP cell checks the worker's MEMRA_SPEC_K=0 dispatch.

use crate::{Engine, cache::Cache, forward::argmax, glm_spec::Glm5SpecKnobs, hybrid::HybridModel};
use memra_gguf::source::SafetensorsSource;
use std::{
    error::Error,
    fs,
    io::Write,
    path::{Path, PathBuf},
};

type Res<T> = Result<T, Box<dyn Error>>;

fn required(key: &str) -> Res<String> {
    Ok(std::env::var(key)?)
}

fn ids(path: &Path) -> Res<Vec<u32>> {
    fs::read_to_string(path)?
        .split_whitespace()
        .map(|x| Ok(x.parse()?))
        .collect()
}

fn write_ids(path: &Path, tokens: &[u32]) -> Res<()> {
    fs::write(
        path,
        tokens
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(" "),
    )?;
    Ok(())
}

fn dump(path: &Path, logits: &[f32]) -> Res<()> {
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "nonfinite target logits"
    );
    let mut f = std::io::BufWriter::new(fs::File::create(path)?);
    for v in logits {
        f.write_all(&v.to_le_bytes())?;
    }
    f.flush()?;
    Ok(())
}

fn load_logits(path: &Path) -> Res<Vec<f32>> {
    let bytes = fs::read(path)?;
    assert_eq!(bytes.len() % 4, 0);
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

fn cache(e: &Engine, m: &HybridModel, prompt: &[u32]) -> Res<Cache> {
    let mut c = Cache::new_planned(e, &m.cfg, &m.plan, prompt.len() + 192)?;
    m.prime_cache(e, prompt, &mut c, 0)?;
    Ok(c)
}

fn plain(e: &Engine, m: &HybridModel, prompt: &[u32]) -> Res<Vec<u32>> {
    let mut c = Cache::new_planned(e, &m.cfg, &m.plan, prompt.len() + 192)?;
    let (logits, _, _) = m.prime_cache(e, prompt, &mut c, 0)?;
    let mut out = vec![argmax(&logits) as u32];
    while out.len() < 160 {
        let logits = m.decode_step(e, *out.last().unwrap(), &mut c)?;
        out.push(argmax(&logits) as u32);
    }
    Ok(out)
}

fn o3(e: &Engine, m: &HybridModel, prompt: &[u32], tape: &[u32]) -> Res<()> {
    let mut serial = cache(e, m, prompt)?;
    let mut batch = cache(e, m, prompt)?;
    let mut offset = 0;
    let mut width = 1;
    while offset < tape.len() {
        let t = width.min(tape.len() - offset);
        let rows = &tape[offset..offset + t];
        let (ll, _, ckpt) = m.glm5_verify_rows(e, rows, &mut batch)?;
        let got = e.dtoh(&ll)?;
        let vocab = got.len() / t;
        assert!(vocab > 0);
        for (r, &token) in rows.iter().enumerate() {
            let expected = m.decode_step(e, token, &mut serial)?;
            assert_eq!(expected.len(), vocab);
            for (v, (&a, &b)) in expected
                .iter()
                .zip(&got[r * vocab..(r + 1) * vocab])
                .enumerate()
            {
                assert!(a.is_finite() && b.is_finite());
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "O3 token={} t={t} vocab={v} serial={a} verify={b}",
                    offset + r
                );
            }
        }
        m.glm5_verify_rollback(e, &mut batch, &ckpt, t)?;
        assert_eq!(batch.pos, serial.pos);
        offset += t;
        width = width % 7 + 1;
    }
    eprintln!("O3 PASS: 160 teacher-forced tokens, widths 1..7, every logit bit-identical");
    Ok(())
}

fn spec(e: &Engine, m: &HybridModel, prompt: &[u32], out: &Path) -> Res<()> {
    let mut metadata = std::io::BufWriter::new(fs::File::create(out.join("rounds.txt"))?);
    let mut accepted = Vec::new();
    let mut round = 0;
    let mut emitted = 1usize; // prompt boundary anchor
    let mut observer = |pos: usize, rows: &[u32], logits: &[f32], j: usize| -> Res<()> {
        dump(&out.join(format!("round-{round}.f32")), logits)?;
        writeln!(
            metadata,
            "{pos} {} {}",
            j + 1,
            rows.iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(" ")
        )?;
        accepted.extend(
            rows[1..=j]
                .iter()
                .copied()
                .take(160usize.saturating_sub(emitted)),
        );
        emitted += j + 1;
        round += 1;
        Ok(())
    };
    let knobs = Glm5SpecKnobs {
        verify_observer: Some(&mut observer),
        ..Default::default()
    };
    let (tokens, drafted, accepted_count) = m.generate_spec_glm5_gated(e, prompt, 160, 6, knobs)?;
    assert_eq!(tokens.len(), 160, "oracle tape stopped early");
    assert!(drafted > 0 && round > 0, "spec oracle did not engage");
    metadata.flush()?;
    write_ids(&out.join("spec.ids"), &tokens)?;
    write_ids(&out.join("accepted.ids"), &accepted)?;
    eprintln!("SPEC K=6 rounds={round} drafted={drafted} accepted={accepted_count}");
    Ok(())
}

/// Softmax of one raw-logit row, in f64 with the max subtracted for stability.
fn softmax(row: &[f32]) -> Vec<f64> {
    let hi = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut p: Vec<f64> = row.iter().map(|&x| (x as f64 - hi).exp()).collect();
    let sum: f64 = p.iter().sum();
    assert!(sum.is_finite() && sum > 0.0, "softmax normaliser {sum}");
    for v in &mut p {
        *v /= sum;
    }
    p
}

/// Worst absolute probability difference between two raw-logit rows.
///
/// O2 compares a TP all-reduce against a PP serial walk of the same rows. Both
/// routes sum the same terms in a different order, so their raw logits differ by
/// reduction noise: on the pinned cell-9 round the median |dlogit| across the
/// 154,880-entry vocabulary is 0.055 to 0.097 and the max is 0.773, on logits
/// whose top entries sit near 27 to 32.
///
/// An elementwise RELATIVE bound on those raw logits is unachievable by
/// construction, not merely tight: any vocabulary entry whose logit lands near
/// zero divides that noise by ~0. On that same round the relative metric reaches
/// max_rel 2422.3 while both rows keep an identical argmax and an identical
/// top-5. Under `check=True` that false FAIL hard-stopped the cell and deleted
/// the whole served phase queued behind it.
///
/// Softmax is what the decoder actually reads and it is well conditioned:
/// probabilities are bounded in [0, 1] and sum to 1, so an ABSOLUTE band on them
/// is scale-free and states the property the oracle means, that the two routes
/// decode the same. The band is a decode-equivalence bound, not a bit-equality
/// bound; O3 is where bit-equality is asserted.
///
/// It is NOT a tight bound, and the reason is a property of the tape rather than
/// of the engine. On a row with a clear winner (p = 0.99999) the same round gives
/// max_dp 4.905e-06. On a row holding a NEAR TIE (p = 0.5783 against 0.4056) the
/// same reduction noise, a dlogit of 0.0247 between the two candidates, moves
/// 1.972e-02 of probability between them, with argmax and top-5 still identical.
/// A near tie is where probability is most fragile and where it matters least,
/// since either candidate is an ordinary sampled outcome. The band therefore has
/// to clear the near-tie rows of the tape, which is why the cell config sets it
/// from a census (`o2-prob-band.tsv`) and not from a guess.
///
/// The other edge of that trade is pinned in the red arm: on a DECIDED row a
/// challenger sitting 8.65 logits below the winner can gain 4 whole logits and
/// still move under 1e-2 of mass. O2 will not see it, and should not, because
/// nothing the sampler does changes. O3 bit-equality is the instrument for
/// movements that small.
fn prob_band_worst(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    softmax(a)
        .into_iter()
        .zip(softmax(b))
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn o2_replay(e: &Engine, m: &HybridModel, prompt: &[u32], tp: &Path, out: &Path) -> Res<()> {
    // Absolute band in PROBABILITY space, set by the cell config from a census of
    // this pair, never from a fixture. The old TP_SPEC_O2_BAND name is deliberately
    // NOT accepted: it meant a relative logit bound, and reusing the name would
    // silently reinterpret a caller's number as a probability.
    let band: f64 = required("TP_SPEC_O2_PROB_BAND")?.parse()?;
    assert!(band.is_finite() && band > 0.0);
    let mut c = cache(e, m, prompt)?;
    let meta = fs::read_to_string(tp.join("rounds.txt"))?;
    assert!(!meta.is_empty());
    // The census is collected for EVERY row and written before anything is
    // asserted. The old band aborted mid-loop on its first offender, so a failing
    // cell left no distribution behind and the next reader had one number and no
    // way to tell a defect from the shape of the tape.
    let mut census: Vec<(usize, usize, f64)> = Vec::new();
    for (round, line) in meta.lines().enumerate() {
        let fields: Vec<usize> = line
            .split_whitespace()
            .map(str::parse)
            .collect::<Result<_, _>>()?;
        let (pos, keep) = (fields[0], fields[1]);
        let rows: Vec<u32> = fields[2..].iter().map(|&x| x as u32).collect();
        assert_eq!(c.pos, pos);
        let (logits, _, ckpt) = m.glm5_verify_rows(e, &rows, &mut c)?;
        let pp = e.dtoh(&logits)?;
        let target = load_logits(&tp.join(format!("round-{round}.f32")))?;
        dump(&out.join(format!("replay-{round}.f32")), &pp)?;
        assert_eq!(pp.len(), target.len());
        let vocab = pp.len() / rows.len();
        for (r, (a, b)) in pp
            .chunks_exact(vocab)
            .zip(target.chunks_exact(vocab))
            .enumerate()
        {
            // Argmax is a hard invariant, not a band: the two routes must pick the
            // same token or the replay is not a replay. It stays an in-loop abort.
            assert_eq!(argmax(a), argmax(b), "O2 argmax round={round} row={r}");
            assert!(
                a.iter().chain(b.iter()).all(|v| v.is_finite()),
                "O2 nonfinite logit round={round} row={r}"
            );
            census.push((round, r, prob_band_worst(a, b)));
        }
        m.glm5_verify_rollback(e, &mut c, &ckpt, keep)?;
    }
    let mut tsv = std::io::BufWriter::new(fs::File::create(out.join("o2-prob-band.tsv"))?);
    writeln!(tsv, "round\trow\tmax_dp")?;
    for &(round, r, dp) in &census {
        writeln!(tsv, "{round}\t{r}\t{dp:.9e}")?;
    }
    tsv.flush()?;
    let &(bad_round, bad_row, worst) = census
        .iter()
        .max_by(|x, y| x.2.total_cmp(&y.2))
        .ok_or("O2 census empty")?;
    assert!(
        worst <= band,
        "O2 prob band round={bad_round} row={bad_row} max_dp={worst} limit={band} \
         over {} rows, census at o2-prob-band.tsv",
        census.len()
    );
    assert_eq!(
        ids(&tp.join("spec.ids"))?,
        ids(&out.join("spec.ids"))?,
        "O2 output sequence"
    );
    assert_eq!(
        ids(&tp.join("accepted.ids"))?,
        ids(&out.join("accepted.ids"))?,
        "O2 accepted-token sequence"
    );
    eprintln!(
        "O2 PASS: actual TP verify rows replayed on PP, argmax equal, max_dp={worst} band={band} (probability space) over {} rows, 160-token tape and accepted sequence equal",
        census.len()
    );
    Ok(())
}

#[test]
#[ignore = "requires the scheduled two-device pair and pinned full model/drafter artifacts"]
fn pair_arm() -> Res<()> {
    let arm = required("TP_SPEC_ARM")?;
    let base = PathBuf::from(required("TP_SPEC_OUT")?);
    let out = base.join(&arm);
    fs::create_dir_all(&out)?;
    let prompt = ids(Path::new(&required("TP_SPEC_PROMPT_IDS")?))?;
    assert!(prompt.len() >= 2);
    let e = Engine::new(0)?;
    // A real second device must exist even on the PP arm. Same-device emulation
    // cannot pass this gate or silently turn it into a single-card check.
    let second = Engine::new(1)?;
    assert_ne!(e.ctx().ordinal(), second.ctx().ordinal());
    drop(second);
    let source = SafetensorsSource::open(Path::new(&required("TP_SPEC_MODEL")?))?;
    let m = HybridModel::load_from_source(&e, &source)?;
    match arm.as_str() {
        "plain" => {
            assert_eq!(m.glm5_tp_rank_count(), Some(2));
            assert!(!crate::glm_spec::glm5_spec_tp_on());
            write_ids(&out.join("plain.ids"), &plain(&e, &m, &prompt)?)?;
        }
        "tp" => {
            assert_eq!(m.glm5_tp_spec_refusal(), None);
            let tape = plain(&e, &m, &prompt)?;
            assert_eq!(
                tape,
                ids(&base.join("plain/plain.ids"))?,
                "O1 engine plain route with spec door ON/OFF"
            );
            write_ids(&out.join("k0.ids"), &tape)?;
            o3(&e, &m, &prompt, &tape)?;
            spec(&e, &m, &prompt, &out)?;
        }
        "pp" => {
            assert!(!m.glm5_has_tp_shards());
            assert_eq!(crate::pp::pp_cuts(m.layers.len()).map(|c| c.len()), Some(3));
            spec(&e, &m, &prompt, &out)?;
            o2_replay(&e, &m, &prompt, &base.join("tp"), &out)?;
        }
        _ => return Err("TP_SPEC_ARM must be plain, tp or pp".into()),
    }
    Ok(())
}

/// Red arm for the O2 probability band.
///
/// These run on any host: the band is a pure function of two logit rows, so the
/// decision the GPU cell makes is exactly the decision under test here. The old
/// elementwise relative-logit metric is reproduced verbatim as `old_rel_worst` so
/// each case can state what BOTH metrics say about the same rows.
#[cfg(test)]
mod o2_band {
    use super::prob_band_worst;

    /// Stand-in for a cell band. The real one comes from the census a run writes
    /// to `o2-prob-band.tsv`; this value clears the near-tie row measured on the
    /// pinned cell-9 round (max_dp 1.972e-02) with headroom, and the tests below
    /// pin what it must accept and what it must still reject.
    const BAND: f64 = 5e-2;
    const VOCAB: usize = 4096;

    /// The metric this repair replaced, kept only so the tests can show its value.
    fn old_rel_worst(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(&reference, &got)| (reference - got).abs() / reference.abs().max(1e-6))
            .fold(0.0, f32::max)
    }

    /// A plausible logit row: one clear winner, a decaying head, a dense tail.
    fn row() -> Vec<f32> {
        (0..VOCAB)
            .map(|i| match i {
                0 => 28.5,
                1..=64 => 20.0 - (i as f32) * 0.15,
                _ => ((i % 37) as f32) * 0.05 - 0.9,
            })
            .collect()
    }

    /// The shape of the row that broke the old band: two candidates almost tied.
    fn near_tie_row() -> Vec<f32> {
        let mut v = row();
        v[0] = 27.2168;
        v[1] = 26.8620;
        v
    }

    fn argmax(v: &[f32]) -> usize {
        v.iter()
            .enumerate()
            .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
            .unwrap()
            .0
    }

    /// GREEN: reduction-order noise on a row with a clear winner. This is the row
    /// the pinned round measured at max_dp 4.905e-06.
    #[test]
    fn accepts_reduction_noise_on_a_decided_row() {
        let a = row();
        let b: Vec<f32> = a
            .iter()
            .enumerate()
            .map(|(i, &v)| v + if i % 2 == 0 { 0.09 } else { -0.09 })
            .collect();
        let dp = prob_band_worst(&a, &b);
        assert!(dp <= BAND, "reduction noise must pass, max_dp={dp}");
    }

    /// GREEN: the same reduction noise on a NEAR TIE moves real probability mass
    /// and must still pass. This is the case a tight band would reject for a
    /// property of the tape rather than a defect in the engine.
    #[test]
    fn accepts_reduction_noise_on_a_near_tie() {
        let a = near_tie_row();
        let mut b = a.clone();
        b[1] += 0.0247; // the dlogit measured between the tied pair
        assert_eq!(argmax(&a), argmax(&b), "the near tie must not flip argmax");
        let dp = prob_band_worst(&a, &b);
        assert!(
            dp > 1e-3,
            "a near tie must move real mass or this case proves nothing, max_dp={dp}"
        );
        assert!(
            dp <= BAND,
            "a near tie must still pass the band, max_dp={dp}"
        );
    }

    /// RED: a real divergence. The runner-up gains 8 logits and comes up level
    /// with the winner, which is two orders past the reduction noise, and the
    /// band must reject it. Argmax is untouched, so the band is the only thing
    /// that can catch it.
    #[test]
    fn rejects_a_real_divergence() {
        let a = row();
        let mut b = a.clone();
        b[1] += 8.0;
        assert_eq!(
            argmax(&a),
            argmax(&b),
            "the divergence must not move argmax"
        );
        let dp = prob_band_worst(&a, &b);
        assert!(
            dp > BAND,
            "an 8.0-logit shift on the runner-up must FAIL the band, max_dp={dp}"
        );
    }

    /// RED: the bound is not vacuously wide. A shift only a little past the band
    /// still fails, so the band has a real edge rather than a rhetorical one. It
    /// is measured on the near-tie row, because that is where a small dlogit
    /// converts into a probability move at all.
    #[test]
    fn rejects_a_divergence_just_past_the_band() {
        let a = near_tie_row();
        let mut b = a.clone();
        b[1] += 0.25;
        let dp = prob_band_worst(&a, &b);
        assert!(
            dp > BAND,
            "a 0.25 shift between tied candidates must FAIL, max_dp={dp}"
        );
        assert!(dp < 2.0 * BAND, "and it must be a NEAR miss, max_dp={dp}");
    }

    /// What this band does NOT catch, pinned so nobody reads it as a tight bound.
    /// On a decided row the runner-up can gain 4 logits, a 53x move in its own
    /// probability, and still shift under 1e-2 of mass, because it started 8.65
    /// logits below the winner. That is the correct answer for a DECODE
    /// equivalence bound: nothing the sampler does changes. O3 bit-equality, not
    /// O2, is the instrument for movements that small.
    #[test]
    fn is_insensitive_to_a_challenger_that_stays_far_below_the_winner() {
        let a = row();
        let mut b = a.clone();
        b[1] += 4.0;
        let dp = prob_band_worst(&a, &b);
        assert!(
            (9.0e-3..1.0e-2).contains(&dp),
            "pin the blind spot, max_dp={dp}"
        );
        assert!(dp <= BAND, "and it passes the band, max_dp={dp}");
    }

    /// The false FAIL that motivated the repair: a tail logit near zero whose
    /// relative error is astronomical and whose probability mass is nil. The old
    /// metric rejected this row; the new one accepts it, and the decode is
    /// identical either way.
    #[test]
    fn accepts_the_row_the_relative_logit_band_rejected() {
        let mut a = row();
        let mut b = row();
        a[VOCAB - 1] = 1e-9;
        b[VOCAB - 1] = 8e-3;
        assert_eq!(argmax(&a), argmax(&b));
        let rel = old_rel_worst(&a, &b);
        assert!(
            rel > 1000.0,
            "the old relative-logit metric must blow up here, max_rel={rel}"
        );
        let dp = prob_band_worst(&a, &b);
        assert!(
            dp <= BAND,
            "the probability band must accept a decode-identical row, max_dp={dp}"
        );
    }

    /// The metric is symmetric and zero on identity, so neither arm is the
    /// privileged one the way `reference.abs()` made it in the old bound.
    #[test]
    fn is_symmetric_and_zero_on_identity() {
        let a = row();
        let mut b = a.clone();
        b[3] += 0.25;
        assert_eq!(prob_band_worst(&a, &a), 0.0);
        assert_eq!(prob_band_worst(&a, &b), prob_band_worst(&b, &a));
        let rel_ab = old_rel_worst(&a, &b);
        let rel_ba = old_rel_worst(&b, &a);
        assert_ne!(
            rel_ab, rel_ba,
            "the old bound was asymmetric, which is why it is gone"
        );
    }

    /// Softmax is computed in f64 with the max subtracted, so a row whose logits
    /// would overflow `exp` in f32 still yields a usable band.
    #[test]
    fn survives_logits_that_would_overflow_f32_exp() {
        let mut a = row();
        let mut b = row();
        for v in a.iter_mut() {
            *v += 200.0;
        }
        for v in b.iter_mut() {
            *v += 200.0;
        }
        b[1] += 8.0;
        let dp = prob_band_worst(&a, &b);
        assert!(dp.is_finite(), "band must stay finite, got {dp}");
        assert!(
            dp > BAND,
            "and must still reject the divergence, max_dp={dp}"
        );
    }
}
