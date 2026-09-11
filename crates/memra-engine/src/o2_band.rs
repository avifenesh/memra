//! The O2 decode-equivalence band: an ABSOLUTE bound in PROBABILITY space.
//!
//! Origin: memra #479, which repaired the band the GLM TP2 pair oracle used, and the
//! GLM TP2 DFlash2 lane (memra #387) that oracle was built to qualify. #387 measured
//! NEGATIVE on the served shape and its feature code is gone; this metric is not, because
//! it is about comparing two reduction orders and has nothing to do with speculation.
//! It is the instrument open issue #487 is written against.
//!
//! The full derivation and the census it is pinned from live in the darklanes lane
//! `research/glm5-o2-band-20260911/LANE.md`.

use crate::forward::argmax;

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
pub fn prob_band_worst(a: &[f32], b: &[f32]) -> f64 {
    row_census(a, b).max_dp
}

/// One replayed row, measured. Everything a verdict needs about the row is taken in
/// one pass, because the row cannot be re-measured once the cell is over: the worst
/// probability move, the token each route picked, and how far apart those two picks
/// sit IN EACH ROUTE'S OWN ROW.
///
/// The last pair is what separates the two findings an argmax disagreement can be. A
/// route that picks a different token because the row holds a near tie has not decoded
/// differently in any sense a sampler would notice: both candidates are ordinary
/// outcomes and each route ranks them within noise of each other. A route that picks a
/// different token on a DECIDED row has diverged, and no band may excuse it.
pub struct O2Row {
    pub max_dp: f64,
    /// What the PP replay picked, and what the TP verify rows say.
    pub pick_replay: usize,
    pub pick_truth: usize,
    /// |p(pick_replay) - p(pick_truth)| inside the replay row and inside the truth row.
    pub gap_replay: f64,
    pub gap_truth: f64,
}

impl O2Row {
    pub fn flipped(&self) -> bool {
        self.pick_replay != self.pick_truth
    }

    /// A flip the band explains: both routes rank the two candidates within the band of
    /// each other, so the disagreement is the tie and not the engine.
    pub fn tie_flip(&self, band: f64) -> bool {
        self.flipped() && self.gap_replay <= band && self.gap_truth <= band
    }
}

/// Measure one replayed row against its truth row: the worst probability move, both
/// picks, and each route's own gap between the two picks.
pub fn row_census(a: &[f32], b: &[f32]) -> O2Row {
    assert_eq!(a.len(), b.len());
    let (pa, pb) = (softmax(a), softmax(b));
    let (ia, ib) = (argmax(a), argmax(b));
    O2Row {
        max_dp: pa
            .iter()
            .zip(&pb)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max),
        pick_replay: ia,
        pick_truth: ib,
        gap_replay: (pa[ia] - pa[ib]).abs(),
        gap_truth: (pb[ia] - pb[ib]).abs(),
    }
}

#[cfg(test)]
mod tests {
    use super::prob_band_worst;

    /// Stand-in for a cell band. The real one comes from the census a run writes
    /// to `o2-prob-band.tsv`; this value clears the near-tie row measured on the
    /// pinned cell-9 round (max_dp 1.972e-02) with headroom, and the tests below
    /// pin what it must accept and what it must still reject.
    const BAND: f64 = 5e-2;
    /// The band the FLIP CLASSIFIER is judged against, and it is NOT `BAND`.
    ///
    /// `BAND` bounds `max_dp`, the worst probability MOVE between two replayed rows.
    /// `tie_flip` asks a different question: whether each route ranks the two disputed
    /// candidates closely enough IN ITS OWN ROW that the disagreement is the tie rather
    /// than the engine. Those are different quantities and they do not share a scale, so
    /// they must not share a constant.
    ///
    /// This value is the one the cells actually ran: `o2_prob_band: 0.5` in the pinned
    /// cell-14 and cell-15 configs. Derived from production, not fitted to the fixture.
    ///
    /// Wiring this to `BAND` is what broke the test below. Under `5e-2` the GREEN case
    /// measures gaps 0.1209 and 0.1060 and the assertion cannot hold for any row anyone
    /// would call a near tie, because a 0.24-logit split between two dominant candidates
    /// already moves 0.12 of probability. The case's own doc says the census row it
    /// stands in for ranked its candidates 0.075 apart, which `5e-2` also rejects, so the
    /// constant was wrong rather than the case. At `5e-1` the GREEN case passes at 0.12
    /// and the RED decided-row case still refuses at 1.00: an 8.3x margin, so the bound
    /// is not vacuous.
    const FLIP_BAND: f64 = 5e-1;
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

    /// GREEN for the flip classifier: the row the pinned cell actually produced.
    /// Round 11 row 1 of the 2026-09-11 census: the replay picked 320 at p=0.4326
    /// while the TP rows picked 22840 at p=0.4296, and each route ranks the two
    /// candidates within 0.075 of each other. Both gaps sit under the band, so the
    /// flip is the tie, and the cell continues.
    #[test]
    fn a_flip_between_two_candidates_the_band_covers_is_a_tie() {
        let mut a = row();
        let mut b = row();
        a[0] = 27.2168;
        a[1] = 27.4602; // the replay's pick is entry 1
        b[0] = 27.4300;
        b[1] = 27.2168; // the truth's pick is entry 0
        let entry = super::row_census(&a, &b);
        assert!(entry.flipped(), "the case must actually flip");
        assert!(
            entry.gap_replay > 0.0 && entry.gap_truth > 0.0,
            "a zero gap would make this vacuous"
        );
        assert!(
            entry.gap_replay < 0.2 && entry.gap_truth < 0.2,
            "the candidates must actually be close, gaps {} and {}",
            entry.gap_replay,
            entry.gap_truth
        );
        assert!(
            entry.tie_flip(FLIP_BAND),
            "gaps {} and {} must read as a tie under band {FLIP_BAND}",
            entry.gap_replay,
            entry.gap_truth
        );
    }

    /// RED for the flip classifier: the same flip on a DECIDED row. The replay picks a
    /// token its own row gives 0.99 while the truth row gives its own pick 0.99, so the
    /// two routes disagree about a token neither row was undecided about. No band may
    /// excuse it, and `tie_flip` must refuse.
    #[test]
    fn a_flip_on_a_decided_row_is_never_a_tie() {
        let mut a = row();
        let mut b = row();
        a[1] = 40.0; // the replay is certain about entry 1
        b[0] = 40.0; // the truth is certain about entry 0
        let entry = super::row_census(&a, &b);
        assert!(entry.flipped(), "the case must actually flip");
        assert!(
            entry.gap_replay > 0.9 && entry.gap_truth > 0.9,
            "both rows must be decided or this proves nothing: {} {}",
            entry.gap_replay,
            entry.gap_truth
        );
        assert!(
            !entry.tie_flip(FLIP_BAND),
            "a decided flip must NOT be excused as a tie, even at the WIDE flip band"
        );
    }

    /// A row with no flip is never a tie flip, whatever its gaps are, so the
    /// classifier cannot report a flip the cell did not have.
    #[test]
    fn an_agreeing_row_is_not_a_flip() {
        let a = row();
        let entry = super::row_census(&a, &a);
        assert!(!entry.flipped());
        assert!(!entry.tie_flip(FLIP_BAND));
        assert_eq!(entry.gap_replay, 0.0);
        assert_eq!(entry.gap_truth, 0.0);
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
