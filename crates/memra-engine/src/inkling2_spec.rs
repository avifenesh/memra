//! Speculative decoding of Inkling 2 Mini: proposals of the DSpark drafter
//! (`crate::inkling2_dspark`) verified by the trunk's route (`crate::inkling2_gpu`).
//!
//! One numeric program per request: every committed token comes out of the route's own forward.
//! A verify call runs the anchor and the proposals as one call of rows; the route computes a row
//! from that row alone, so each row's logits are the bits a one-row decode call would give. On a
//! partial accept the route goes back to the convolution history saved before the call and runs
//! the accepted rows again (the same rows, so the same bits); the K/V caches need no copy (see
//! `Carry`). Greedy output with drafting therefore equals greedy decode without it, token for
//! token, which `ink2-spec --check-plain` gates.

use crate::inkling2_dspark::{DraftGpu, Shared};
use crate::inkling2_gpu::{Head, Route};

/// How proposals and the committed tokens are chosen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sampling {
    /// Temperature 0: proposals are the drafter's argmax; a proposal is accepted when it equals
    /// the trunk's argmax.
    Greedy,
    /// Probabilistic drafting: proposals are samples of the drafter's distribution at
    /// `temperature`, verified by the probability-ratio test against the trunk's distribution
    /// at the same temperature (residual resample on the first rejection, a bonus sample after
    /// a full accept).
    Sample { temperature: f64, seed: u64 },
}

/// Counts of one generation.
#[derive(Debug, Clone, Default)]
pub struct SpecStats {
    /// Verify calls.
    pub steps: usize,
    /// Proposals sent to verification.
    pub proposed: usize,
    /// Proposals accepted.
    pub accepted: usize,
    /// `hist[n]`: verify calls that accepted `n` proposals.
    pub hist: Vec<usize>,
    /// Verify calls followed by a replay of the accepted rows.
    pub replays: usize,
    pub prefill_s: f64,
    pub decode_s: f64,
    /// Committed tokens after the prompt.
    pub generated: usize,
}

impl SpecStats {
    /// Mean accepted proposals per verify call.
    pub fn acceptance_length(&self) -> f64 {
        self.accepted as f64 / self.steps.max(1) as f64
    }

    /// Mean committed tokens per verify call (accepted proposals plus the trunk's own token).
    pub fn tokens_per_step(&self) -> f64 {
        (self.accepted + self.steps) as f64 / self.steps.max(1) as f64
    }
}

/// splitmix64: the sampler's generator, seeded per request.
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// `softmax(logits / temperature)` in f64.
pub fn probs(logits: &[f32], temperature: f64) -> Vec<f64> {
    let max = logits.iter().fold(f32::NEG_INFINITY, |m, v| m.max(*v)) as f64;
    let e: Vec<f64> = logits
        .iter()
        .map(|v| ((*v as f64 - max) / temperature).exp())
        .collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

/// One draw from a distribution (inverse CDF; the last index takes the rounding remainder).
pub fn draw(p: &[f64], rng: &mut Rng) -> u32 {
    let total: f64 = p.iter().sum();
    let u = rng.unit() * total;
    let mut acc = 0.0;
    for (i, v) in p.iter().enumerate() {
        acc += v;
        if u < acc {
            return i as u32;
        }
    }
    (p.len() - 1) as u32
}

/// Greedy decode without drafting: the prompt in calls of `route.chunk` rows, then one row per
/// call. Stops after `n` tokens or at a token of `stop`.
pub fn plain_greedy(
    route: &mut Route,
    prompt: &[u32],
    n: usize,
    stop: &[u32],
) -> Result<Vec<u32>, String> {
    route.reset()?;
    let first = route.call(prompt, Head::LastArgmax, None)?;
    let mut tok = *first.argmax.last().ok_or("no argmax")?;
    let mut out = vec![tok];
    while out.len() < n && !stop.contains(&tok) {
        tok = route.forward(&[tok], Head::LastArgmax, None)?.argmax[0];
        out.push(tok);
    }
    Ok(out)
}

/// Speculative decode: the prompt (its rows committed to the drafter), then verify calls of the
/// anchor plus `k` proposals (1..=7). Stops after `n` tokens or at a token of `stop`.
#[allow(clippy::too_many_arguments)] // allow: one request's whole configuration
pub fn generate(
    route: &mut Route,
    drafter: &mut DraftGpu,
    prompt: &[u32],
    n: usize,
    k: usize,
    sampling: Sampling,
    stop: &[u32],
) -> Result<(Vec<u32>, SpecStats), String> {
    generate_arm(route, drafter, prompt, n, k, sampling, stop, false)
}

/// [`generate`], or with `skip_restore` its red arm: on a partial accept the rejected rows stay
/// in the convolution history and only the position goes back. The gate must catch it.
#[allow(clippy::too_many_arguments)] // allow: one request's whole configuration
pub fn generate_arm(
    route: &mut Route,
    drafter: &mut DraftGpu,
    prompt: &[u32],
    n: usize,
    k: usize,
    sampling: Sampling,
    stop: &[u32],
    skip_restore: bool,
) -> Result<(Vec<u32>, SpecStats), String> {
    let taps = route
        .plan()
        .inkling2
        .as_ref()
        .ok_or("not an inkling2 plan")?
        .drafter
        .taps
        .clone();
    let slots: Vec<u32> = taps
        .iter()
        .filter_map(|t| match t {
            memra_gguf::model_plan::DrafterTap::SlotInput { slot } => Some(*slot),
            memra_gguf::model_plan::DrafterTap::FinalNorm => None,
        })
        .collect();
    if slots.len() + 1 != taps.len()
        || taps.last() != Some(&memra_gguf::model_plan::DrafterTap::FinalNorm)
    {
        return Err("drafter taps other than slot inputs then the final norm".into());
    }
    if route.chunk < k + 1 {
        return Err(format!(
            "a verify of {} rows; the route holds {}",
            k + 1,
            route.chunk
        ));
    }
    route.set_capture(&slots)?;
    route.reset()?;
    drafter.reset();
    let shared = Shared::of_route(route);
    let chunk = route.chunk;
    let mut rng = match sampling {
        Sampling::Sample { seed, .. } => Rng(seed),
        Sampling::Greedy => Rng(0),
    };
    let mut stats = SpecStats {
        hist: vec![0; k + 1],
        ..SpecStats::default()
    };
    let t0 = std::time::Instant::now();
    // prefill: the route's chunking, every row committed to the drafter
    let mut last_logits = Vec::new();
    let mut anchor = 0;
    // (a sampled request runs its last prompt row alone, for its logits; rows do not depend on
    // the chunking)
    let mut calls: Vec<&[u32]> = prompt.chunks(chunk).collect();
    if matches!(sampling, Sampling::Sample { .. }) {
        let tail = calls.pop().ok_or("an empty prompt")?;
        if tail.len() > 1 {
            calls.push(&tail[..tail.len() - 1]);
        }
        calls.push(&tail[tail.len() - 1..]);
    }
    let count = calls.len();
    for (i, part) in calls.into_iter().enumerate() {
        let last = i + 1 == count;
        let head = match (last, sampling) {
            (false, _) => Head::None,
            (true, Sampling::Greedy) => Head::LastArgmax,
            (true, Sampling::Sample { .. }) => Head::All,
        };
        let out = route.forward(part, head, None)?;
        let cap = route.captured().ok_or("capture is off")?;
        drafter.commit(&route.dev, cap, chunk, 0, part.len())?;
        if last {
            anchor = *out.argmax.last().ok_or("no argmax")?;
            if let Sampling::Sample { temperature, .. } = sampling {
                last_logits = out.logits;
                anchor = draw(&probs(&last_logits, temperature), &mut rng);
            }
        }
    }
    route.dev.sync()?;
    stats.prefill_s = t0.elapsed().as_secs_f64();
    let _ = last_logits;
    let t1 = std::time::Instant::now();
    let mut carry = route.carry()?;
    let mut out = vec![anchor];
    let vocab = route.vocab();
    while out.len() < n && !stop.contains(&anchor) {
        let kk = k.min(n - out.len()).max(1);
        let (drafts, qrows) = match sampling {
            Sampling::Greedy => (
                drafter.propose(&route.dev, shared, anchor, kk, None)?,
                Vec::new(),
            ),
            Sampling::Sample { temperature, .. } => {
                let mut q = Vec::new();
                let mut pick = |_: usize, logits: &[f32]| {
                    let pr = probs(logits, temperature);
                    let t = draw(&pr, &mut rng);
                    q.push(pr);
                    t
                };
                let d = drafter.propose(&route.dev, shared, anchor, kk, Some(&mut pick))?;
                (d, q)
            }
        };
        let mut rows = vec![anchor];
        rows.extend_from_slice(&drafts);
        route.save_carry(&mut carry)?;
        let head = match sampling {
            Sampling::Greedy => Head::AllArgmax,
            Sampling::Sample { .. } => Head::All,
        };
        let verify = route.forward(&rows, head, None)?;
        // accepted proposals and the trunk's own next token
        let (accepted, next) = match sampling {
            Sampling::Greedy => {
                let mut a = 0;
                while a < kk && drafts[a] == verify.argmax[a] {
                    a += 1;
                }
                (a, verify.argmax[a])
            }
            Sampling::Sample { temperature, .. } => {
                let mut a = 0;
                let mut next = None;
                while a < kk {
                    let p = probs(&verify.logits[a * vocab..(a + 1) * vocab], temperature);
                    let q = &qrows[a];
                    let d = drafts[a] as usize;
                    let ratio = if q[d] > 0.0 { p[d] / q[d] } else { 1.0 };
                    if rng.unit() < ratio.min(1.0) {
                        a += 1;
                        continue;
                    }
                    let residual: Vec<f64> =
                        p.iter().zip(q).map(|(x, y)| (x - y).max(0.0)).collect();
                    next = Some(if residual.iter().sum::<f64>() > 0.0 {
                        draw(&residual, &mut rng)
                    } else {
                        draw(&p, &mut rng)
                    });
                    break;
                }
                let next = match next {
                    Some(t) => t,
                    None => draw(
                        &probs(&verify.logits[kk * vocab..(kk + 1) * vocab], temperature),
                        &mut rng,
                    ),
                };
                (a, next)
            }
        };
        if accepted < kk && skip_restore {
            route.position = carry.position() + accepted as u64 + 1;
        } else if accepted < kk {
            // back to the history before the verify, and run the committed rows again
            route.restore_carry(&carry)?;
            route.forward(&rows[..accepted + 1], Head::None, None)?;
            stats.replays += 1;
        }
        let cap = route.captured().ok_or("capture is off")?;
        drafter.commit(&route.dev, cap, chunk, 0, accepted + 1)?;
        stats.steps += 1;
        stats.proposed += kk;
        stats.accepted += accepted;
        stats.hist[accepted] += 1;
        for &t in &drafts[..accepted] {
            out.push(t);
            if stop.contains(&t) {
                break;
            }
        }
        if out.last().is_some_and(|t| stop.contains(t)) {
            break;
        }
        anchor = next;
        out.push(anchor);
    }
    route.dev.sync()?;
    stats.decode_s = t1.elapsed().as_secs_f64();
    out.truncate(n);
    if let Some(at) = out.iter().position(|t| stop.contains(t)) {
        out.truncate(at + 1);
    }
    stats.generated = out.len();
    route.set_capture(&[])?;
    Ok((out, stats))
}
