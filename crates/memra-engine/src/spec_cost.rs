//! Speculative-decoding cost calibration (lane/batch-spec step 4).
//!
//! The serving spec gate decides, per model and per concurrency, whether speculative rounds or
//! plain batched decode produce more tokens per second. That decision is a property of the device
//! and the model (GB10 is bandwidth-bound and verifies up to 8 rows for the price of one; a 5090 on
//! a 9B is closer to compute-bound), so the gate prices it from tables measured on the serving
//! box at boot rather than from thresholds measured elsewhere:
//!
//! - `verify_ms[R-1]`: one target verify forward over R rows of one session (R = 1..=16), the
//!   t-parallel walk every pooled and serial MTP verify runs;
//! - `plain_ms[B-1]`: one plain batched decode step over B sessions (B = 1..=P, P the model's
//!   plain wave cap: 16, or 32 on the b32 tier), the serving tick's lean device-sampled step;
//! - `round_overhead_ms`: what a speculative round costs a session beyond its verify forward
//!   (drafting, accept walk, commit, host syncs), from a real session burst.
//!
//! Each table is measured at one fixed depth with the engine in the serving numeric class. Rows
//! not measured directly are linearly interpolated between measured neighbours.

use crate::Engine;
use crate::cache::Cache;
use crate::decode_batch::DevSamp;
use crate::hybrid::HybridModel;
use std::time::Instant;

/// Measured costs for one model on one device (see the module docs).
#[derive(Clone, Debug, PartialEq)]
pub struct SpecCostTable {
    /// KV depth the tables were measured at.
    pub depth: usize,
    /// Verify forward ms for R = 1..=16 rows (index R-1).
    pub verify_ms: Vec<f64>,
    /// Plain batched decode step ms for B = 1..=P sessions (index B-1), P = the plain wave cap.
    pub plain_ms: Vec<f64>,
    /// Per-session, per-round speculative cost beyond the verify forward (one session).
    pub round_overhead_ms: f64,
    /// The same per-session cost measured on a pooled lockstep cohort of `.0` sessions
    /// (batched draft, shared verify), when the model pools. Between 1 and `.0` sessions the
    /// gate interpolates; past `.0` it holds the cohort figure.
    pub pooled_overhead_ms: Option<(usize, f64)>,
}

impl SpecCostTable {
    /// Per-session round overhead at `n` lockstep sessions.
    pub fn overhead_at(&self, n: usize) -> f64 {
        match self.pooled_overhead_ms {
            Some((m, o)) if n >= m => o,
            Some((m, o)) if n > 1 => {
                self.round_overhead_ms
                    + (o - self.round_overhead_ms) * (n - 1) as f64 / (m - 1) as f64
            }
            _ => self.round_overhead_ms,
        }
    }
}

/// Cohort size the pooled overhead is measured at: four sessions at K=3 fill exactly one
/// 16-row pooled verify.
pub const POOLED_OVERHEAD_COHORT: usize = 4;

/// Rows and batch widths measured directly; the rest interpolate.
const VERIFY_POINTS: [usize; 8] = [1, 2, 3, 4, 6, 8, 12, 16];
const PLAIN_POINTS: [usize; 6] = [1, 2, 4, 8, 16, 32];
const REPS: usize = 3;

/// Fill `1..=16` from measured `(x, ms)` points by linear interpolation.
pub fn interpolate_16(points: &[(usize, f64)]) -> Vec<f64> {
    interpolate_n(points, 16)
}

/// Fill `1..=n` from measured `(x, ms)` points by linear interpolation.
pub fn interpolate_n(points: &[(usize, f64)], n: usize) -> Vec<f64> {
    (1..=n)
        .map(|x| {
            if let Some(&(_, v)) = points.iter().find(|(px, _)| *px == x) {
                return v;
            }
            let lo = points.iter().rev().find(|(px, _)| *px < x).copied();
            let hi = points.iter().find(|(px, _)| *px > x).copied();
            match (lo, hi) {
                (Some((x0, y0)), Some((x1, y1))) => {
                    y0 + (y1 - y0) * (x - x0) as f64 / (x1 - x0) as f64
                }
                (Some((_, y)), None) | (None, Some((_, y))) => y,
                (None, None) => f64::NAN,
            }
        })
        .collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Fit check shared by the boot probes that prime `target` equal sessions at `depth` (a cache
/// plus its snapshot each). Call it after session `i` (0-based) is primed and snapshotted. At
/// `i == 0` it records the pool's USED bytes; the first session also carries one-time scratch,
/// so at `i == 1` the delta is one session's footprint, and the probe refuses when the other
/// `target - 2` need more than is allocatable (driver free, page-cache aware on unified memory,
/// plus the pool's reusable cache). It stops only a probe whose priming could not complete. The
/// sync lands the prime's scratch frees first; a pool that cannot be queried reads (0, 0) and
/// the probe runs.
fn probe_session_fit(
    e: &Engine,
    i: usize,
    target: usize,
    depth: usize,
    used_after_first: &mut usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if i > 1 {
        return Ok(());
    }
    e.stream().synchronize()?;
    let (reserved, used) = e.pool_reserved_used();
    if i == 0 {
        *used_after_first = used;
        return Ok(());
    }
    let (free, _) = crate::unified_memory::allocatable_mem_info(e.ctx())?;
    let available = free.saturating_add(reserved.saturating_sub(used));
    match sessions_fit_refusal(
        target,
        depth,
        used.saturating_sub(*used_after_first),
        available,
    ) {
        Some(why) => Err(why.into()),
        None => Ok(()),
    }
}

/// The decision of [`probe_session_fit`]: `Some(reason)` when the sessions after the first
/// two need more than `available` bytes.
fn sessions_fit_refusal(
    target: usize,
    depth: usize,
    per_session: usize,
    available: usize,
) -> Option<String> {
    let others = target.saturating_sub(2);
    let needed = per_session.saturating_mul(others);
    (needed > available).then(|| {
        format!(
            "{target} sessions at depth {depth} do not fit: the other {others} need {}MB \
             ({}MB each), {}MB allocatable",
            needed >> 20,
            per_session >> 20,
            available >> 20,
        )
    })
}

impl HybridModel {
    /// Boot probe for the 32-row decode wave (batch-spec 2026-11-03): one plain step over 32
    /// sessions in one wave against the same 32 sessions as two 16-row waves, at `depth`,
    /// median of 3, the single wave timed on the dp4a and on the tensor-core b32 tier (both
    /// bit-identical). Returns `(two_waves_ms, one_wave_ms, one_wave_tc_ms)`. The one-wave tier is per-row
    /// bit-identical to the two-wave program, so the faster form is a pure speed choice, and
    /// the margin is small and device-shaped (GB10 27B: one wave 3.8% faster; an RTX 5090
    /// laptop on a 9B: within its thermal drift), hence measured on the serving box. Every
    /// buffer is dropped on return; the freed blocks stay cached in the async pool, so the
    /// caller hands them back to the driver.
    ///
    /// The probe sizes itself to the device: after the second primed session it measures one
    /// session's pool footprint (cache plus snapshot; the first session also carries one-time
    /// scratch) and refuses when the remaining sessions need more than is allocatable, instead
    /// of priming until the driver runs out (RTX 5090, Qwen3.8-27B: 48.5 s to the OOM).
    pub fn decode_wave_probe(
        &self,
        e: &Engine,
        depth: usize,
    ) -> Result<(f64, f64, f64), Box<dyn std::error::Error>> {
        const N: usize = 32;
        let vocab = (self.cfg.n_vocab as usize).max(64) - 16;
        let mut caches: Vec<Cache> = Vec::with_capacity(N);
        let mut snaps = Vec::with_capacity(N);
        let mut used_after_first = 0usize;
        for i in 0..N {
            let mut c = crate::pp::new_cache(e, &self.cfg, depth + 32)?;
            let p: Vec<u32> = (0..depth as u32)
                .map(|j| (55 + (i as u32 + 1) * 97 + j * 31) % vocab as u32)
                .collect();
            let _ = self.prime_cache(e, &p, &mut c, 0)?;
            snaps.push(c.snapshot(e)?);
            caches.push(c);
            probe_session_fit(e, i, N, depth, &mut used_after_first)?;
        }
        let toks: Vec<u32> = (0..N as u32)
            .map(|j| (4242 + j * 7) % vocab as u32)
            .collect();
        let samp = vec![Some(DevSamp::new(0.0, 0, 0, 0, 1.0, 0.0)); N];
        let masks = vec![None; N];
        let (mut two, mut one, mut one_tc) = (
            Vec::with_capacity(REPS),
            Vec::with_capacity(REPS),
            Vec::with_capacity(REPS),
        );
        let tc_before = e.mmvq_tc_on();
        // arms: two 16-row waves, one 32-row wave on the dp4a tier, one on the tensor-core tier
        for rep in 0..=REPS {
            for arm in 0..3 {
                let wide = arm > 0;
                e.set_tc_b32(arm == 2);
                for (c, s) in caches.iter_mut().zip(&snaps) {
                    c.rollback(e, s, 0)?;
                }
                let t0 = Instant::now();
                if wide {
                    let mut refs: Vec<&mut Cache> = caches.iter_mut().collect();
                    let _ = self.decode_step_batch_sampled_lean_masked(
                        e, &toks, &mut refs, &samp, &masks, true,
                    )?;
                } else {
                    let (lo, hi) = caches.split_at_mut(N / 2);
                    for (half, off) in [(lo, 0usize), (hi, N / 2)] {
                        let mut refs: Vec<&mut Cache> = half.iter_mut().collect();
                        let _ = self.decode_step_batch_sampled_lean_masked(
                            e,
                            &toks[off..off + N / 2],
                            &mut refs,
                            &samp[off..off + N / 2],
                            &masks[off..off + N / 2],
                            true,
                        )?;
                    }
                }
                if rep > 0 {
                    let ms = t0.elapsed().as_secs_f64() * 1e3;
                    match arm {
                        0 => two.push(ms),
                        1 => one.push(ms),
                        _ => one_tc.push(ms),
                    }
                }
            }
        }
        e.set_tc_b32(tc_before);
        Ok((median(two), median(one), median(one_tc)))
    }

    /// Measure this model's [`SpecCostTable`] at `depth` (see the module docs). Runs on the
    /// serving engine before readiness; every buffer it allocates is dropped on return. The
    /// plain-batch table primes up to 16 (or 32) sessions and checks their fit after the second,
    /// like the wave probe: on an RTX 5090 a Qwen3.8-27B calibration primed until the driver ran
    /// out (46 s) to fail anyway.
    pub fn spec_cost_calibrate(
        &self,
        e: &Engine,
        depth: usize,
        plain_cap: usize,
    ) -> Result<SpecCostTable, Box<dyn std::error::Error>> {
        let plain_cap = plain_cap.clamp(1, 32);
        let plain_points: Vec<usize> = PLAIN_POINTS
            .iter()
            .copied()
            .filter(|&b| b <= plain_cap.max(16))
            .collect();
        let vocab = (self.cfg.n_vocab as usize).max(64) - 16;
        let n_vocab = self.cfg.n_vocab as usize;
        let prompt = |salt: u32| -> Vec<u32> {
            (0..depth as u32)
                .map(|j| (55 + salt * 97 + j * 31) % vocab as u32)
                .collect()
        };
        // ---- verify forward over R rows ----
        let mut verify = Vec::new();
        {
            let mut c = crate::pp::new_cache(e, &self.cfg, depth + 32)?;
            let _ = self.prime_cache(e, &prompt(0), &mut c, 0)?;
            let snap = c.snapshot(e)?;
            let _exact = e.exact_scope(true);
            for &r in &VERIFY_POINTS {
                let toks: Vec<u32> = (0..r as u32)
                    .map(|j| (977 + j * 131) % vocab as u32)
                    .collect();
                let mut ms = Vec::with_capacity(REPS);
                for rep in 0..=REPS {
                    c.rollback(e, &snap, 0)?;
                    let t0 = Instant::now();
                    let (logits, _h) =
                        self.decode_step_t_h_emb_dev(e, &toks, depth, &mut c, None)?;
                    let am = e.argmax_token_device(&logits, n_vocab)?;
                    let _ = e.dtoh_u32_one(&am)?;
                    if rep > 0 {
                        ms.push(t0.elapsed().as_secs_f64() * 1e3);
                    }
                }
                verify.push((r, median(ms)));
            }
        }
        // ---- plain batched decode over B sessions ----
        let mut plain = Vec::new();
        {
            let b_max = *plain_points.last().unwrap();
            let mut caches: Vec<Cache> = Vec::with_capacity(b_max);
            let mut snaps = Vec::with_capacity(b_max);
            let mut used_after_first = 0usize;
            for i in 0..b_max {
                let mut c = crate::pp::new_cache(e, &self.cfg, depth + 32)?;
                let _ = self.prime_cache(e, &prompt(i as u32 + 1), &mut c, 0)?;
                snaps.push(c.snapshot(e)?);
                caches.push(c);
                probe_session_fit(e, i, b_max, depth, &mut used_after_first)?;
            }
            for &b in &plain_points {
                let toks: Vec<u32> = (0..b as u32)
                    .map(|j| (4242 + j * 7) % vocab as u32)
                    .collect();
                let samp = vec![Some(DevSamp::new(0.0, 0, 0, 0, 1.0, 0.0)); b];
                let masks = vec![None; b];
                let mut ms = Vec::with_capacity(REPS);
                for rep in 0..=REPS {
                    for (c, s) in caches.iter_mut().zip(&snaps).take(b) {
                        c.rollback(e, s, 0)?;
                    }
                    let mut refs: Vec<&mut Cache> = caches.iter_mut().take(b).collect();
                    let t0 = Instant::now();
                    let _ = self.decode_step_batch_sampled_lean_masked(
                        e, &toks, &mut refs, &samp, &masks, true,
                    )?;
                    if rep > 0 {
                        ms.push(t0.elapsed().as_secs_f64() * 1e3);
                    }
                }
                plain.push((b, median(ms)));
            }
        }
        let verify_ms = interpolate_16(&verify);
        let plain_ms = interpolate_n(&plain, plain_cap.max(16));
        // ---- per-round overhead beyond the verify, from a real session burst ----
        let k = 3usize;
        let mut sess = self.new_session(e, depth + 160)?;
        let _ = self.generate_spec_session_sampled(e, &mut sess, &prompt(99), 8, k, None, None)?;
        let before = sess.telemetry();
        let t0 = Instant::now();
        let _ = self.generate_spec_session_sampled(e, &mut sess, &[], 96, k, None, None)?;
        let wall = t0.elapsed().as_secs_f64() * 1e3;
        let rounds = sess.telemetry().delta_since(&before).rounds.max(1) as f64;
        let round_overhead_ms = (wall / rounds - verify_ms[k]).max(0.0);
        // ---- the same on a pooled lockstep cohort ----
        let pooled_overhead_ms = if self.spec_pool_capable() {
            let m = POOLED_OVERHEAD_COHORT;
            let mut sessions = Vec::with_capacity(m);
            for i in 0..m {
                let mut s = self.new_session(e, depth + 160)?;
                let _ = self.generate_spec_session_sampled(
                    e,
                    &mut s,
                    &prompt(200 + i as u32),
                    8,
                    k,
                    None,
                    None,
                )?;
                sessions.push(s);
            }
            let before: Vec<_> = sessions.iter().map(|s| s.telemetry()).collect();
            let reqs = sessions
                .iter_mut()
                .map(|sess| crate::spec::PooledBurstReq {
                    sess,
                    max_new: 96,
                    k,
                    sampling: None,
                    on_commit: None,
                })
                .collect();
            let t0 = Instant::now();
            for r in self.generate_spec_sessions_pooled(e, reqs) {
                r?;
            }
            let wall = t0.elapsed().as_secs_f64() * 1e3;
            let rounds = sessions
                .iter()
                .zip(&before)
                .map(|(s, b)| s.telemetry().delta_since(b).rounds)
                .max()
                .unwrap_or(1)
                .max(1) as f64;
            let rows = (m * (k + 1)).min(16);
            let per_session = (wall / rounds - verify_ms[rows - 1]) / m as f64;
            Some((m, per_session.max(0.0)))
        } else {
            None
        };
        Ok(SpecCostTable {
            depth,
            verify_ms,
            plain_ms,
            round_overhead_ms,
            pooled_overhead_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    /// Both boot probes check their fit between their last prime and their first timed batch
    /// step. On an RTX 5090 (24 GB) with Qwen3.8-27B the wave probe primed 32 sessions until
    /// the driver ran out (48.5 s) and the spec-cost calibration 16 (46 s), both to fail anyway.
    /// Anchored on comment-stripped, whitespace-free production text.
    #[test]
    fn boot_probes_check_their_fit_before_timing() {
        let src = include_str!("spec_cost.rs");
        let code: String = src
            .lines()
            .map(|l| l.split("//").next().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n");
        let prod = &code[..code.find("\nmod tests").expect("tests module exists")];
        let flat: String = prod.chars().filter(|c| !c.is_whitespace()).collect();
        for probe in ["pubfndecode_wave_probe(", "pubfnspec_cost_calibrate("] {
            let start = flat.find(probe).expect(probe);
            let rest = &flat[start + probe.len()..];
            let body = &rest[..rest.find("pubfn").unwrap_or(rest.len())];
            let timed = body
                .find("self.decode_step_batch_sampled_lean_masked(")
                .expect("timed batch step");
            let prime = body[..timed]
                .rfind("self.prime_cache(")
                .expect("priming loop");
            let fit = body[prime..timed]
                .find("probe_session_fit(")
                .unwrap_or_else(|| panic!("{probe} checks its fit after priming, before timing"));
            assert!(prime + fit < timed);
        }
    }

    /// The fit decision refuses only when the sessions after the first two cannot fit, and
    /// names what they need against what is allocatable.
    #[test]
    fn probe_fit_refuses_only_past_what_is_allocatable() {
        const MB: usize = 1 << 20;
        assert_eq!(
            super::sessions_fit_refusal(32, 2048, 505 * MB, 30 * 505 * MB),
            None
        );
        let why = super::sessions_fit_refusal(32, 2048, 505 * MB, 30 * 505 * MB - 1)
            .expect("one byte short refuses");
        assert_eq!(
            why,
            "32 sessions at depth 2048 do not fit: the other 30 need 15150MB (505MB each), \
             15149MB allocatable"
        );
        assert!(super::sessions_fit_refusal(16, 2048, 505 * MB, 7016 * MB).is_some());
        assert_eq!(super::sessions_fit_refusal(2, 2048, usize::MAX, 0), None);
        assert_eq!(super::sessions_fit_refusal(16, 2048, 0, 0), None);
    }

    #[test]
    fn pooled_overhead_interpolates_from_one_session_to_the_cohort() {
        let t = super::SpecCostTable {
            depth: 2048,
            verify_ms: vec![1.0; 16],
            plain_ms: vec![1.0; 16],
            round_overhead_ms: 4.0,
            pooled_overhead_ms: Some((4, 1.0)),
        };
        assert_eq!(t.overhead_at(1), 4.0);
        assert_eq!(t.overhead_at(2), 3.0);
        assert_eq!(t.overhead_at(4), 1.0);
        assert_eq!(t.overhead_at(9), 1.0);
        let serial_only = super::SpecCostTable {
            pooled_overhead_ms: None,
            ..t
        };
        assert_eq!(serial_only.overhead_at(8), 4.0);
    }

    #[test]
    fn interpolation_keeps_measured_points_and_fills_between() {
        let v = super::interpolate_16(&[(1, 10.0), (4, 16.0), (16, 40.0)]);
        assert_eq!(v.len(), 16);
        assert_eq!(v[0], 10.0);
        assert_eq!(v[1], 12.0);
        assert_eq!(v[3], 16.0);
        assert_eq!(v[9], 28.0);
        assert_eq!(v[15], 40.0);
    }
}
