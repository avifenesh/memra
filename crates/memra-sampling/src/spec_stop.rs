//! Draft-chain confidence stop for speculative decoding (`MEMRA_SPEC_PMIN`, `MEMRA_SPEC_PMIN0`)
//! and the rule that keeps a SAMPLED speculative round exact (memra#673, memra#412).
//!
//! A drafter proposes `x_j ~ q_j` slot by slot. The confidence gate ends the chain early when
//! the drafter is unsure, and the target then verifies the kept prefix by rejection sampling
//! against the same `q_j` (accept with `min(1, p/q)`, on rejection draw from
//! `norm(max(0, p - q))`, after a full accept draw the bonus from `p`).
//!
//! THE RULE. The stop decision at slot `j` must be a function of what the chain knew BEFORE
//! the draw at `j` (the context and the drafted prefix `x_<j`), never of `x_j` itself. Then
//! the chain length is a stopping time of the prefix, and at every position the emitted
//! token is exactly `p`: either the slot is verified (the standard rejection argument) or the
//! round ends there and the bonus is a fresh draw from `p`. The statistic this crate names is
//! [`row_confidence`], the max probability of the slot's distribution.
//!
//! THE DEFECT IT REPLACES. The gate used to threshold the probability of the token it had
//! just drawn and discard that token when it fell below `p_min`. Keeping `x_j` only when
//! `q(x_j) >= p_min` changes the proposal to `q` censored to its confident tokens, while
//! verify still divides by the uncensored `q`. Two-token counterexamples, both reproduced
//! exactly by the tests below: `p = q = (0.8, 0.2)` with B censored emits `(0.96, 0.04)`
//! (#673), and `p = (0.5, 0.5)`, `q = (0.9, 0.1)`, `p_min = 0.7` after an accepted prefix
//! emits `(0.55, 0.45)` (#412).
//!
//! Greedy drafts are unchanged by the rule: an argmax pick's probability IS the row max.
//! `p_min <= 0` keeps every slot, so the unarmed program is untouched.
//!
//! This module is pure host code so the rule and its exactness proof run in the CUDA-free
//! unit suite. The engine's drafting loops (memra-engine `spec.rs`, `glm_spec.rs`) call
//! [`stops_at`] and [`conf_keep`]; the device side computes [`row_confidence`] with the
//! argmax + prob-of-token kernels.

/// The p-min break predicate every drafting loop applies at slot `j`: stop when the gate is
/// armed (`p_min > 0`) and `conf < p_min`, where slot 0 may stop only when `slot0_armed`
/// (`MEMRA_SPEC_PMIN0`, plus whatever the caller requires for a legal zero-draft round).
/// `conf` MUST be draw-independent ([`row_confidence`] of the slot's row); see the module doc.
/// Strict `<`: a slot whose confidence equals `p_min` is kept.
#[inline]
pub fn stops_at(conf: f32, j: usize, p_min: f32, slot0_armed: bool) -> bool {
    p_min > 0.0 && conf < p_min && (j > 0 || slot0_armed)
}

/// Keep the longest prefix of per-slot confidences whose every slot clears `p_min` (the
/// [`stops_at`] break applied to a precomputed confidence vector). Slot 0 survives a miss
/// unless `pmin0`. `p_min <= 0` keeps all slots. Prefix truncation is forced by the accept
/// rule anyway: a kept slot after a dropped one could never commit.
pub fn conf_keep(conf: &[f32], p_min: f32, pmin0: bool) -> usize {
    if p_min <= 0.0 {
        return conf.len();
    }
    conf.iter()
        .enumerate()
        .take_while(|&(j, &c)| !stops_at(c, j, p_min, pmin0))
        .count()
}

/// Draw-independent slot confidence: the max probability of the slot's distribution.
/// For a greedy draft it equals the probability of the drafted token. Empty row = 0.
pub fn row_confidence(row: &[f32]) -> f32 {
    row.iter().copied().fold(0.0, f32::max)
}

/// [`row_confidence`] of each `width`-wide row of `rows`, first `n` rows (a DFlash2 selector
/// records its per-slot candidate distributions as one flat `[n x top_k]` buffer).
///
/// # Panics
/// When `width == 0` or `rows` holds fewer than `n` rows.
pub fn row_confidences(rows: &[f32], width: usize, n: usize) -> Vec<f32> {
    assert!(width > 0, "row_confidences: zero row width");
    assert!(
        rows.len() / width >= n,
        "row_confidences: {} values hold fewer than {n} rows of {width}",
        rows.len()
    );
    rows.chunks_exact(width)
        .take(n)
        .map(row_confidence)
        .collect()
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
// allow: the enumeration indexes p, q, r and the residual by the same vocab id
mod tests {
    //! Exact small-vocabulary enumeration of one sampled speculative round followed by plain
    //! target sampling, compared to the target's autoregressive law in exact rationals.
    use super::*;
    use std::collections::BTreeMap;

    // ---- exact rationals (i128, reduced, overflow-checked) ----
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Q(i128, i128); // num/den, den > 0, reduced

    fn gcd(a: i128, b: i128) -> i128 {
        let (mut a, mut b) = (a.abs(), b.abs());
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    }
    impl Q {
        fn new(n: i128, d: i128) -> Q {
            assert!(d != 0);
            let s = if d < 0 { -1 } else { 1 };
            let g = gcd(n, d).max(1);
            Q(s * n / g, s * d / g)
        }
        const ZERO: Q = Q(0, 1);
        const ONE: Q = Q(1, 1);
        fn add(self, o: Q) -> Q {
            let g = gcd(self.1, o.1);
            let l = (self.1 / g).checked_mul(o.1).expect("den overflow");
            let n = self
                .0
                .checked_mul(l / self.1)
                .and_then(|a| o.0.checked_mul(l / o.1).and_then(|b| a.checked_add(b)))
                .expect("num overflow");
            Q::new(n, l)
        }
        fn sub(self, o: Q) -> Q {
            self.add(Q(-o.0, o.1))
        }
        fn mul(self, o: Q) -> Q {
            let g1 = gcd(self.0, o.1).max(1);
            let g2 = gcd(o.0, self.1).max(1);
            let n = (self.0 / g1).checked_mul(o.0 / g2).expect("num overflow");
            let d = (self.1 / g2).checked_mul(o.1 / g1).expect("den overflow");
            Q::new(n, d)
        }
        fn div(self, o: Q) -> Q {
            assert!(o.0 != 0, "division by zero");
            self.mul(Q::new(o.1, o.0))
        }
        fn min(self, o: Q) -> Q {
            if self.le(o) { self } else { o }
        }
        fn le(self, o: Q) -> bool {
            self.0 * o.1 <= o.0 * self.1
        }
        fn is_pos(self) -> bool {
            self.0 > 0
        }
        fn f32(self) -> f32 {
            (self.0 as f64 / self.1 as f64) as f32
        }
    }
    fn dist(w: &[i128]) -> Vec<Q> {
        let s: i128 = w.iter().sum();
        w.iter().map(|&x| Q::new(x, s)).collect()
    }

    /// What the gate thresholds at a slot.
    #[derive(Clone, Copy, Debug)]
    enum Stat {
        /// The FIXED rule: `row_confidence` of the confidence row (draw-independent).
        RowMax,
        /// The OLD rule (red twin): the confidence row's probability of the drawn token.
        Chosen,
    }

    /// One model: target law `p(ctx)`, proposal `q(ctx)` (what the drafter draws from and
    /// what verify divides by), and the confidence row `r(ctx)` the gate reads (spec.rs: the
    /// head's raw T=1 softmax while `q` is the filtered/tempered row; DFlash2: `r = q`).
    struct Model<'a> {
        vocab: usize,
        p: &'a dyn Fn(&[usize]) -> Vec<Q>,
        q: &'a dyn Fn(&[usize]) -> Vec<Q>,
        r: &'a dyn Fn(&[usize]) -> Vec<Q>,
    }

    #[derive(Clone, Copy, Debug)]
    struct Gate {
        k: usize,
        p_min: f32,
        slot0_armed: bool,
        stat: Stat,
    }

    type Law = BTreeMap<Vec<usize>, Q>;

    thread_local! {
        /// Stop branches taken on this thread (non-vacuity: the exhaustive test proves the
        /// row-max gate actually fires, so its exactness is not the unarmed program's).
        static STOPS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    fn put(out: &mut Law, seq: &[usize], w: Q) {
        let e = out.entry(seq.to_vec()).or_insert(Q::ZERO);
        *e = e.add(w);
    }

    /// Plain target sampling from `seq` until the horizon.
    fn tail(m: &Model, seq: &mut Vec<usize>, horizon: usize, w: Q, out: &mut Law) {
        if seq.len() == horizon {
            put(out, seq, w);
            return;
        }
        let p = (m.p)(seq);
        for y in 0..m.vocab {
            if p[y].is_pos() {
                seq.push(y);
                tail(m, seq, horizon, w.mul(p[y]), out);
                seq.pop();
            }
        }
    }

    /// One speculative round from `seq` at draft slot `j`, every earlier draft of the round
    /// accepted (drafting slot j after verifying slots < j has the same law as drafting the
    /// whole chain first: later draws never touch an earlier verdict). Then plain sampling.
    fn round(
        m: &Model,
        g: Gate,
        seq: &mut Vec<usize>,
        j: usize,
        horizon: usize,
        w: Q,
        out: &mut Law,
    ) {
        if seq.len() == horizon {
            put(out, seq, w); // the rest of the round marginalizes to w
            return;
        }
        let p = (m.p)(seq);
        if j == g.k {
            // full accept: bonus from p, then plain sampling
            for y in 0..m.vocab {
                if p[y].is_pos() {
                    seq.push(y);
                    tail(m, seq, horizon, w.mul(p[y]), out);
                    seq.pop();
                }
            }
            return;
        }
        let q = (m.q)(seq);
        let r = (m.r)(seq);
        let r32: Vec<f32> = r.iter().map(|v| v.f32()).collect();
        // residual law norm(max(0, p - q)) (only reached when some x has q(x) > p(x))
        let pos: Vec<Q> = (0..m.vocab)
            .map(|y| {
                let d = p[y].sub(q[y]);
                if d.is_pos() { d } else { Q::ZERO }
            })
            .collect();
        let mass = pos.iter().fold(Q::ZERO, |a, &b| a.add(b));
        for x in 0..m.vocab {
            if !q[x].is_pos() {
                continue;
            }
            let conf = match g.stat {
                Stat::RowMax => row_confidence(&r32),
                Stat::Chosen => r32[x],
            };
            let wx = w.mul(q[x]);
            if stops_at(conf, j, g.p_min, g.slot0_armed) {
                STOPS.with(|c| c.set(c.get() + 1));
                // x discarded; the round ends on the bonus drawn from p at this position
                for y in 0..m.vocab {
                    if p[y].is_pos() {
                        seq.push(y);
                        tail(m, seq, horizon, wx.mul(p[y]), out);
                        seq.pop();
                    }
                }
                continue;
            }
            // accept with min(1, p/q)
            let acc = Q::ONE.min(p[x].div(q[x]));
            if acc.is_pos() {
                seq.push(x);
                round(m, g, seq, j + 1, horizon, wx.mul(acc), out);
                seq.pop();
            }
            let rej = Q::ONE.sub(acc);
            if rej.is_pos() {
                for y in 0..m.vocab {
                    if pos[y].is_pos() {
                        seq.push(y);
                        tail(m, seq, horizon, wx.mul(rej).mul(pos[y].div(mass)), out);
                        seq.pop();
                    }
                }
            }
        }
    }

    fn emitted(m: &Model, g: Gate, horizon: usize) -> Law {
        let mut out = Law::new();
        round(m, g, &mut Vec::new(), 0, horizon, Q::ONE, &mut out);
        out
    }

    fn target(m: &Model, horizon: usize) -> Law {
        let mut out = Law::new();
        tail(m, &mut Vec::new(), horizon, Q::ONE, &mut out);
        out
    }

    /// Marginal law of position `i` (0-based) of an emitted law.
    fn marginal(law: &Law, i: usize, vocab: usize) -> Vec<Q> {
        let mut m = vec![Q::ZERO; vocab];
        for (s, &w) in law {
            m[s[i]] = m[s[i]].add(w);
        }
        m
    }

    fn eq_law(a: &Law, b: &Law) -> bool {
        let keys: std::collections::BTreeSet<_> = a.keys().chain(b.keys()).collect();
        keys.into_iter()
            .all(|k| a.get(k).copied().unwrap_or(Q::ZERO) == b.get(k).copied().unwrap_or(Q::ZERO))
    }

    fn total(law: &Law) -> Q {
        law.values().fold(Q::ZERO, |a, &b| a.add(b))
    }

    // ---- the #673 counterexample: p = q = (0.8, 0.2), B censored at an eligible slot ----

    fn ab_fixed(_: &[usize]) -> Vec<Q> {
        dist(&[4, 1])
    }

    /// Slot 0 armed (PMIN0, the zero-draft arm): the first emitted token.
    #[test]
    fn mtp_counterexample_slot0_old_rule_emits_096_004_and_row_max_emits_p() {
        let m = Model {
            vocab: 2,
            p: &ab_fixed,
            q: &ab_fixed,
            r: &ab_fixed,
        };
        let old = Gate {
            k: 1,
            p_min: 0.5,
            slot0_armed: true,
            stat: Stat::Chosen,
        };
        let law = emitted(&m, old, 1);
        assert_eq!(marginal(&law, 0, 2), vec![Q::new(24, 25), Q::new(1, 25)]);
        let new = Gate {
            stat: Stat::RowMax,
            ..old
        };
        assert!(eq_law(&emitted(&m, new, 1), &target(&m, 1)));
        assert!(eq_law(&emitted(&m, new, 3), &target(&m, 3)));
    }

    /// Slot 1 (after an accepted slot 0, no PMIN0): the second emitted token is biased the
    /// same way under the old rule, conditional on any first token.
    #[test]
    fn mtp_counterexample_after_accepted_prefix() {
        let m = Model {
            vocab: 2,
            p: &ab_fixed,
            q: &ab_fixed,
            r: &ab_fixed,
        };
        let old = Gate {
            k: 2,
            p_min: 0.5,
            slot0_armed: false,
            stat: Stat::Chosen,
        };
        let law = emitted(&m, old, 2);
        // p = q at slot 0: slot 0 always accepts, so position 1 carries the censoring bias
        assert_eq!(marginal(&law, 0, 2), vec![Q::new(4, 5), Q::new(1, 5)]);
        assert_eq!(marginal(&law, 1, 2), vec![Q::new(24, 25), Q::new(1, 25)]);
        let new = Gate {
            stat: Stat::RowMax,
            ..old
        };
        assert!(eq_law(&emitted(&m, new, 3), &target(&m, 3)));
    }

    // ---- the #412 counterexample: after an accepted prefix, p = (.5,.5), q = (.9,.1) ----

    fn g412_p(ctx: &[usize]) -> Vec<Q> {
        if ctx.is_empty() {
            dist(&[1, 0])
        } else {
            dist(&[1, 1])
        }
    }
    fn g412_q(ctx: &[usize]) -> Vec<Q> {
        if ctx.is_empty() {
            dist(&[1, 0])
        } else {
            dist(&[9, 1])
        }
    }

    #[test]
    fn dflash2_counterexample_old_rule_emits_055_045_and_row_max_emits_p() {
        // DFlash2: the gate reads the selector's own recorded row (r = q)
        let m = Model {
            vocab: 2,
            p: &g412_p,
            q: &g412_q,
            r: &g412_q,
        };
        let old = Gate {
            k: 2,
            p_min: 0.7,
            slot0_armed: false,
            stat: Stat::Chosen,
        };
        let law = emitted(&m, old, 2);
        assert_eq!(marginal(&law, 1, 2), vec![Q::new(11, 20), Q::new(9, 20)]);
        let new = Gate {
            stat: Stat::RowMax,
            ..old
        };
        let fixed = emitted(&m, new, 2);
        assert_eq!(marginal(&fixed, 1, 2), vec![Q::new(1, 2), Q::new(1, 2)]);
        assert!(eq_law(
            &fixed,
            &target(&m, 3).into_iter().fold(Law::new(), |mut a, (s, w)| {
                put(&mut a, &s[..2], w);
                a
            })
        ));
        assert!(eq_law(&emitted(&m, new, 3), &target(&m, 3)));
    }

    // ---- exhaustive family: every cutoff / accept / residual / bonus path together ----

    /// Deterministic context-dependent tables: small integer weights from a hash of the
    /// context, so p, q and r differ by prefix and exercise every branch.
    fn table(ctx: &[usize], salt: u64, vocab: usize, zero_ok: bool) -> Vec<Q> {
        let mut h: u64 = 0x9E37_79B9_7F4A_7C15 ^ salt;
        for &t in ctx {
            h = h.wrapping_mul(0x100_0000_01B3).wrapping_add(t as u64 + 1);
            h ^= h >> 29;
        }
        h ^= (ctx.len() as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        let mut w = Vec::with_capacity(vocab);
        for i in 0..vocab {
            h = h
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407 + i as u64);
            let lo = if zero_ok { 0 } else { 1 };
            w.push(lo + ((h >> 33) % 6) as i128);
        }
        if w.iter().all(|&x| x == 0) {
            w[0] = 1;
        }
        dist(&w)
    }

    #[test]
    fn row_max_rule_is_exact_over_every_small_configuration() {
        let mut red_seen = 0usize;
        let mut fixed_fired = 0usize;
        let mut cases = 0usize;
        for vocab in [2usize, 3] {
            for salt in 0..6u64 {
                let p = move |c: &[usize]| table(c, salt * 3 + 1, vocab, salt % 2 == 1);
                let q = move |c: &[usize]| table(c, salt * 3 + 2, vocab, salt % 3 == 2);
                let r = move |c: &[usize]| table(c, salt * 3 + 3, vocab, false);
                // spec.rs shape: gate on a separate raw row r; DFlash2 shape: r = q
                let separate = Model {
                    vocab,
                    p: &p,
                    q: &q,
                    r: &r,
                };
                let selfrow = Model {
                    vocab,
                    p: &p,
                    q: &q,
                    r: &q,
                };
                for m in [&separate, &selfrow] {
                    for k in 1..=3usize {
                        let horizon = k + 1;
                        let want = target(m, horizon);
                        assert_eq!(total(&want), Q::ONE);
                        for p_min in [0.0f32, 0.2, 0.35, 0.5, 0.7, 0.95] {
                            for slot0_armed in [false, true] {
                                let g = Gate {
                                    k,
                                    p_min,
                                    slot0_armed,
                                    stat: Stat::RowMax,
                                };
                                let before = STOPS.with(|c| c.get());
                                let got = emitted(m, g, horizon);
                                if STOPS.with(|c| c.get()) > before {
                                    fixed_fired += 1;
                                }
                                assert_eq!(total(&got), Q::ONE);
                                assert!(
                                    eq_law(&got, &want),
                                    "row-max rule biased: vocab={vocab} salt={salt} k={k} \
                                     p_min={p_min} slot0={slot0_armed}"
                                );
                                let old = emitted(
                                    m,
                                    Gate {
                                        stat: Stat::Chosen,
                                        ..g
                                    },
                                    horizon,
                                );
                                assert_eq!(total(&old), Q::ONE);
                                if p_min == 0.0 {
                                    // unarmed gate: the two rules are the same program
                                    assert!(eq_law(&old, &want));
                                } else if !eq_law(&old, &want) {
                                    red_seen += 1;
                                }
                                cases += 1;
                            }
                        }
                    }
                }
            }
        }
        // the red twin: the old chosen-token rule is biased on a large share of armed cases
        assert!(
            red_seen * 4 > cases,
            "old rule biased on only {red_seen}/{cases} cases"
        );
        // the fixed gate truncated chains in a large share of cases and stayed exact
        assert!(
            fixed_fired * 4 > cases,
            "row-max gate fired on only {fixed_fired}/{cases}"
        );
    }

    /// Greedy drafts: the pick is the argmax, so the chosen-token statistic equals the row
    /// max and the fix changes nothing on the greedy program.
    #[test]
    fn greedy_pick_confidence_equals_row_confidence() {
        let rows: [&[f32]; 4] = [&[0.1, 0.7, 0.2], &[0.5, 0.5], &[1.0], &[0.3, 0.3, 0.4]];
        for row in rows {
            let am = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0)))
                .map(|(i, _)| i)
                .unwrap();
            assert_eq!(row[am], row_confidence(row));
        }
    }

    #[test]
    fn stop_predicate_and_prefix_contracts() {
        // strict <, slot-0 exemption, unarmed gate
        assert!(!stops_at(0.5, 1, 0.5, false));
        assert!(stops_at(0.49, 1, 0.5, false));
        assert!(!stops_at(0.1, 0, 0.5, false));
        assert!(stops_at(0.1, 0, 0.5, true));
        assert!(!stops_at(0.0, 3, 0.0, true));
        assert_eq!(conf_keep(&[0.9, 0.2, 0.9], 0.5, false), 1);
        assert_eq!(conf_keep(&[0.2, 0.9], 0.5, false), 2);
        assert_eq!(conf_keep(&[0.2, 0.9], 0.5, true), 0);
        assert_eq!(conf_keep(&[0.2, 0.1], 0.0, true), 2);
        assert_eq!(conf_keep(&[], 0.5, true), 0);
        // the archived DFlash2 selector cases (research/glm5-dflash-rootcause-20260909)
        let rows = [0.5, 0.5, 0.9, 0.1, 0.6, 0.4, 0.9, 0.1];
        assert_eq!(conf_keep(&row_confidences(&rows, 2, 4), 0.7, false), 2);
        assert_eq!(conf_keep(&row_confidences(&rows, 2, 4), 0.7, true), 0);
        assert_eq!(conf_keep(&row_confidences(&rows, 2, 4), 0.0, true), 4);
        assert_eq!(conf_keep(&row_confidences(&rows, 2, 1), 0.7, false), 1);
    }
}
