//! Host-side sampler chain (BASE-2, MEMRA-BUILD-MAP §BASE-2). Ports llama.cpp CPU sampler
//! semantics (llama-sampler.cpp): repetition/freq/presence penalties -> temperature -> top-k ->
//! top-p -> min-p -> categorical draw. Greedy (temp<=0) = argmax, the bit-exact reference.
//!
//! Runs on the host over the full [n_vocab] f32 logit vector already brought back by the per-step
//! D2H sync (decode.rs) — at B=2-4 this is single-µs, no GPU kernel needed (the GPU-fused sampler
//! is a deferred PERF item, only needed once CUDA-graph removes the D2H barrier).

use std::collections::HashMap;

/// Sampler configuration. Defaults = greedy (temp 0). Order of application matches llama.cpp.
#[derive(Clone, Debug)]
pub struct SamplerConfig {
    pub temperature: f32,      // <= 0.0 => greedy argmax (penalties/top-k/p ignored)
    pub top_k: usize,          // 0 => disabled (keep all)
    pub top_p: f32,            // 1.0 => disabled
    pub min_p: f32,            // 0.0 => disabled
    pub penalty_last_n: usize, // window of recent tokens for penalties (0 => disabled)
    pub penalty_repeat: f32,   // 1.0 => disabled (llama default 1.0)
    pub penalty_freq: f32,     // 0.0 => disabled
    pub penalty_present: f32,  // 0.0 => disabled
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        SamplerConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            penalty_last_n: 0,
            penalty_repeat: 1.0,
            penalty_freq: 0.0,
            penalty_present: 0.0,
            seed: 0,
        }
    }
}

/// SESSION-RESUME SAMPLER IDENTITY (lane/session-resume-sampler-predicate-20260820; receipts
/// `research/spec-cache-20260818/SESSION-RESUME-PREDICATE.md`).
///
/// The canonical form of a request's sampler, for exactly one question: **may this request resume
/// a parked whole session that some OTHER request's sampler shaped?** The spec pool's resume probe
/// compared prompts and never samplers — that omission is how a filtered request inherited a draft
/// graph captured unfiltered (`memra-engine` `SampledGraphKey`, lane/graph-s-key-exactness-
/// 20260819). Keying the graph closed the exactness hole; it did not make cross-sampler resume
/// SOUND, and the house posture in that situation is refuse-on-ambiguity with the refusal naming
/// itself. This type is that predicate.
///
/// CANONICALIZATION, and why each rule is safe. Two encodings that name the same program must
/// compare equal, or the predicate refuses resumes that cost nothing to allow:
/// - `temperature <= 0.0` is GREEDY — `-1.0` and `0.0` are one program, so `temp_bits` is pinned
///   to `0.0` in that regime and `greedy` carries the distinction. The greedy/sampled flip is
///   itself a refusal: the two arms consume a parked `next_pred`/`pending_tok` differently and
///   engage different captured draft graphs.
/// - `top_k == 0`, `top_p >= 1.0`, `min_p <= 0.0` are each the OFF sentinel (matching
///   `is_spec_sampling` and `SampledGraphKey::pure_temp`), canonicalized so `top_p 1.5` and
///   `top_p 1.0` do not look like a change.
/// - Penalties are OFF as a group iff `penalty_last_n == 0` or all three coefficients are neutral
///   — the same `pen_on` predicate `spec.rs` computes. Off canonicalizes to the whole disabled
///   tuple, so `penalty_last_n 64` with neutral coefficients equals penalties absent.
///
/// Float fields compare by BITS after canonicalization (no NaN/`-0.0` surprise), the same
/// discipline `SampledGraphKey` uses.
///
/// `seed` IS carried and DELIBERATELY NOT COMPARED — see [`SamplerIdentity::mismatch`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SamplerIdentity {
    greedy: bool,
    temp_bits: u32,
    /// Carried for the record and for callers that want to log it; NOT part of `mismatch`.
    seed: u64,
    top_k: usize,
    top_p_bits: u32,
    min_p_bits: u32,
    penalty_last_n: usize,
    penalty_repeat_bits: u32,
    penalty_freq_bits: u32,
    penalty_present_bits: u32,
}

impl SamplerIdentity {
    /// Canonical identity of a sampler configuration.
    pub fn of(cfg: &SamplerConfig) -> Self {
        let greedy = cfg.temperature <= 0.0;
        let pen_on = cfg.penalty_last_n > 0
            && (cfg.penalty_repeat != 1.0 || cfg.penalty_freq != 0.0 || cfg.penalty_present != 0.0);
        SamplerIdentity {
            greedy,
            temp_bits: if greedy { 0.0f32 } else { cfg.temperature }.to_bits(),
            seed: cfg.seed,
            top_k: cfg.top_k,
            top_p_bits: if cfg.top_p >= 1.0 { 1.0f32 } else { cfg.top_p }.to_bits(),
            min_p_bits: if cfg.min_p <= 0.0 { 0.0f32 } else { cfg.min_p }.to_bits(),
            penalty_last_n: if pen_on { cfg.penalty_last_n } else { 0 },
            penalty_repeat_bits: if pen_on { cfg.penalty_repeat } else { 1.0f32 }.to_bits(),
            penalty_freq_bits: if pen_on { cfg.penalty_freq } else { 0.0f32 }.to_bits(),
            penalty_present_bits: if pen_on { cfg.penalty_present } else { 0.0f32 }.to_bits(),
        }
    }

    /// The seed this identity was built from (logging/receipts only — never compared).
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The FIRST field on which `self` (an incoming request) differs from `parked` (the sampler
    /// that shaped a parked session), as a stable name for the refusal line — `None` when the two
    /// samplers are equivalent and the resume is legal. A refusal that does not say why is
    /// indistinguishable from an unwired mechanism, so the name is the deliverable, not a nicety.
    ///
    /// Order is fixed and coarsest-first (`regime` before the field that only exists inside one
    /// regime), so the reported name is the most informative one rather than an artifact of struct
    /// layout.
    ///
    /// **`seed` IS NOT COMPARED, deliberately.** It is the one sampler field a resume may change,
    /// for two reasons that are both mechanical:
    /// - The only parked state that BAKES the seed is the sampled draft graph, and
    ///   `SampledGraphKey` already carries `seed`: a seed change drops the parked graph and
    ///   recaptures. (`memra-engine` `spec.rs`; pinned by `seed_alone_still_rekeys_the_draft_graph`
    ///   in that crate's `sampled_graph_key` tests.)
    /// - The session's persisted Philox counters (`SpecSession::sctr/uctr`) are counter-based:
    ///   `philox(seed', ctr)` continued from another seed's counter position is an independent
    ///   stream, not a repeated one. Reproducibility is already scoped per `(seed, session)`
    ///   rather than per seed (`memra-server` `worker.rs`, the spec-burst sampling note), so a
    ///   changed seed costs nothing that same-seed resume was not already costing.
    ///
    /// Comparing it would refuse essentially ALL sampled traffic: omitting `seed` on a serve
    /// request draws fresh per-request entropy, so every turn of every seed-omitting conversation
    /// would carry a "changed" seed. That is a cost with no soundness gain, which is exactly the
    /// trade this predicate exists to make explicitly rather than by accident.
    pub fn mismatch(&self, parked: &Self) -> Option<&'static str> {
        if self.greedy != parked.greedy {
            return Some("regime");
        }
        if self.temp_bits != parked.temp_bits {
            return Some("temperature");
        }
        if self.top_k != parked.top_k {
            return Some("top_k");
        }
        if self.top_p_bits != parked.top_p_bits {
            return Some("top_p");
        }
        if self.min_p_bits != parked.min_p_bits {
            return Some("min_p");
        }
        if self.penalty_last_n != parked.penalty_last_n {
            return Some("penalty_last_n");
        }
        if self.penalty_repeat_bits != parked.penalty_repeat_bits {
            return Some("penalty_repeat");
        }
        if self.penalty_freq_bits != parked.penalty_freq_bits {
            return Some("penalty_freq");
        }
        if self.penalty_present_bits != parked.penalty_present_bits {
            return Some("penalty_present");
        }
        None
    }

    /// THE PRE-LANE PREDICATE, RESTATED (teeth, not production). The spec pool-resume probe
    /// applied no sampler test at all — it compared prompts and nothing else — so every sampler
    /// pair was admitted. Restating it here keeps the refusal tests DECISIVE instead of
    /// tautological: the same pair that `mismatch` names must be admitted by this, or the test is
    /// asserting against a mechanism that never existed.
    ///
    /// It is also what `MEMRA_SPEC_RESUME_SAMPLER=0` selects at runtime (the rollback door and the
    /// A/B arm the cost measurement needs), so this function is the single definition of "legacy"
    /// for both the tests and the server.
    pub fn legacy_admits(&self, _parked: &Self) -> bool {
        true
    }
}

/// Stateful sampler: owns the RNG + the recent-token history (for penalties).
pub struct Sampler {
    cfg: SamplerConfig,
    rng: SplitMix64,
    history: Vec<u32>, // recently emitted tokens (for penalty window)
    // Counts over exactly the active penalty window. Keeping this incrementally makes the host
    // oracle cheaper too, and lets the serving path upload O(unique ids) sparse penalty state
    // instead of either the full vocabulary or an O(history^2) device-side dedup walk.
    penalty_counts: HashMap<u32, u32>,
}

impl Sampler {
    pub fn new(cfg: SamplerConfig) -> Self {
        let rng = SplitMix64::new(cfg.seed);
        Sampler {
            cfg,
            rng,
            history: Vec::new(),
            penalty_counts: HashMap::new(),
        }
    }

    pub fn is_greedy(&self) -> bool {
        self.cfg.temperature <= 0.0
    }
    /// Sampled spec in its FASTEST regime: pure temperature, no truncation filters, no
    /// penalties. Filters and penalties are also distribution-exact under the rejection
    /// verify (spec.rs applies both symmetrically to draft q and target p), so they remain
    /// spec-ELIGIBLE — see `spec_eligible` in memra-server's worker, the authoritative
    /// predicate. What they cost is the in-graph draft chain: the captured sampled draft
    /// samples from the RAW softmax and can hold neither per-row filter stats nor a varying
    /// penalty history, so `spec.rs` engages `graph_s` only in this pure-temp regime
    /// (`pure_temp`) and otherwise falls back to the eager draft chain. This predicate names
    /// that regime; it is NOT an eligibility test.
    pub fn is_spec_sampling(&self) -> bool {
        self.cfg.temperature > 0.0
            && self.cfg.penalty_repeat == 1.0
            && self.cfg.penalty_freq == 0.0
            && self.cfg.penalty_present == 0.0
            && self.cfg.top_k == 0
            && self.cfg.top_p >= 1.0
            && self.cfg.min_p <= 0.0
    }
    pub fn top_k(&self) -> usize {
        self.cfg.top_k
    }
    pub fn penalty_last_n(&self) -> usize {
        self.cfg.penalty_last_n
    }
    pub fn penalty_repeat(&self) -> f32 {
        self.cfg.penalty_repeat
    }
    pub fn penalty_freq(&self) -> f32 {
        self.cfg.penalty_freq
    }
    pub fn penalty_present(&self) -> f32 {
        self.cfg.penalty_present
    }
    pub fn top_p(&self) -> f32 {
        self.cfg.top_p
    }
    pub fn min_p(&self) -> f32 {
        self.cfg.min_p
    }
    pub fn temperature(&self) -> f32 {
        self.cfg.temperature
    }
    pub fn seed(&self) -> u64 {
        self.cfg.seed
    }
    /// This sampler's canonical [`SamplerIdentity`] — the whole-session resume predicate's input.
    pub fn identity(&self) -> SamplerIdentity {
        SamplerIdentity::of(&self.cfg)
    }

    fn penalties_on(&self) -> bool {
        self.cfg.penalty_last_n > 0
            && (self.cfg.penalty_repeat != 1.0
                || self.cfg.penalty_freq != 0.0
                || self.cfg.penalty_present != 0.0)
    }

    /// Sparse `(token_id, count)` rows for the current penalty window. The order cannot affect
    /// the arithmetic because every entry mutates one distinct logit; avoiding a per-token sort
    /// is material on long agent histories.
    pub fn penalty_counts(&self) -> Vec<(u32, u32)> {
        debug_assert!(self.penalty_counts.values().all(|&count| count > 0));
        self.penalty_counts
            .iter()
            .map(|(&id, &n)| (id, n))
            .collect()
    }

    /// Record an emitted token so subsequent penalties see it.
    pub fn accept(&mut self, token: u32) {
        if self.penalties_on() {
            let n = self.cfg.penalty_last_n;
            if self.history.len() >= n {
                let expired = self.history[self.history.len() - n];
                let remove = {
                    let count = self
                        .penalty_counts
                        .get_mut(&expired)
                        .expect("active penalty window lost an accepted token");
                    *count -= 1;
                    *count == 0
                };
                if remove {
                    self.penalty_counts.remove(&expired);
                }
            }
            *self.penalty_counts.entry(token).or_insert(0) += 1;
        }
        self.history.push(token);
    }

    /// Sample the next token id from raw logits [n_vocab]. Does NOT mutate logits in place beyond
    /// a local copy. Returns the chosen token id. (Caller should `accept()` it afterwards.)
    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        // Greedy fast path: argmax over RAW logits (penalties don't change the argmax direction
        // enough to matter for the reference path; llama greedy is also pre-penalty argmax only
        // when no penalties set — but to stay correct under penalties we still apply them first).
        if self.is_greedy()
            && self.cfg.penalty_repeat == 1.0
            && self.cfg.penalty_freq == 0.0
            && self.cfg.penalty_present == 0.0
        {
            return argmax_u32(logits);
        }

        // Work on (id, logit) candidates.
        let mut cand: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| (i as u32, l))
            .collect();

        // 1. Penalties (operate on logits, over the last-n history window).
        // `cand` is DENSE and INDEX-ALIGNED here by construction (built from
        // `logits.iter().enumerate()` immediately above, nothing has filtered it yet), so the
        // penalty pass indexes straight into it instead of hashing every candidate. See
        // `apply_penalties_dense`.
        self.apply_penalties_dense(&mut cand);

        // Greedy-with-penalties: argmax after penalties, no sampling.
        if self.is_greedy() {
            let mut best = cand[0];
            for &c in &cand[1..] {
                if c.1 > best.1 {
                    best = c;
                }
            }
            return best.0;
        }

        // 2. Temperature scale.
        if self.cfg.temperature > 0.0 && self.cfg.temperature != 1.0 {
            let inv = 1.0 / self.cfg.temperature;
            for c in cand.iter_mut() {
                c.1 *= inv;
            }
        }

        // 3. top-k: keep the k highest-logit candidates (partial sort by logit desc).
        if self.cfg.top_k > 0 && self.cfg.top_k < cand.len() {
            cand.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            cand.truncate(self.cfg.top_k);
        }

        // softmax over the surviving candidates (numerically stable).
        softmax_inplace(&mut cand);

        // 4. top-p (nucleus): smallest set whose cumulative prob >= top_p. Needs desc-by-prob order.
        if self.cfg.top_p < 1.0 {
            cand.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            let mut cum = 0.0f32;
            let mut keep = 0usize;
            for (i, c) in cand.iter().enumerate() {
                cum += c.1;
                keep = i + 1;
                if cum >= self.cfg.top_p {
                    break;
                }
            }
            cand.truncate(keep.max(1));
        }

        // 5. min-p: keep candidates with prob >= min_p * max_prob.
        if self.cfg.min_p > 0.0 {
            let maxp = cand.iter().map(|c| c.1).fold(0.0f32, f32::max);
            let thresh = self.cfg.min_p * maxp;
            cand.retain(|c| c.1 >= thresh);
            if cand.is_empty() {
                return argmax_u32(logits);
            } // safety
        }

        // renormalize the surviving probs and draw.
        let sum: f32 = cand.iter().map(|c| c.1).sum();
        let r = self.rng.next_f32() * sum;
        let mut acc = 0.0f32;
        for c in &cand {
            acc += c.1;
            if acc >= r {
                return c.0;
            }
        }
        cand.last().unwrap().0
    }

    /// llama.cpp penalty over a DENSE, INDEX-ALIGNED candidate slice: `cand[i].0 == i`.
    ///
    /// Same arithmetic as `apply_penalties_scan_reference`, on exactly the same elements, in the
    /// same order — but O(distinct penalized tokens) instead of O(n_vocab) hash lookups.
    ///
    /// WHY (lane/glm5-host-audit, 2026-09-01). The scan form did one `HashMap<u32,u32>` SipHash
    /// probe PER CANDIDATE, i.e. one per vocabulary entry: ~152k probes per token on the Qwen
    /// class. `penalty_counts` holds at most `PEN_WINDOW_MAX` distinct ids and in practice the
    /// number of distinct generated tokens so far, so the loop was inverted the expensive way
    /// round. This matters in production, not in theory: `devsample_meta` refuses the device
    /// sampler for any penalized config unless `MEMRA_SERVE_DEVPENALTY=1`, the whole fleet runs
    /// it at 0, and a served model whose VENDOR-RECOMMENDED non-thinking arm carries
    /// `presence_penalty` therefore lands every token of every request on this function.
    ///
    /// BIT-IDENTICAL BY CONSTRUCTION, and gated as such rather than asserted in prose: the set of
    /// touched entries is identical (`penalty_counts` covers exactly the window, which the
    /// debug_assert below re-checks), the per-entry arithmetic is copied unchanged, and each
    /// entry is touched exactly once in both forms, so no float re-association is possible.
    /// `apply_penalties_scan_reference` is kept as the ORACLE and
    /// `dense_penalties_match_the_scan_reference_bitwise` compares them over randomized inputs.
    ///
    /// THE HOST ORACLE FOR DEVICE PENALTIES (lane/spec-exclusions-20260902): the memra-engine
    /// `penalize_logits*` kernels (`cu/spec_sample.cu`, `keskar_penalize_rn`) are written to
    /// produce these exact f32 bits for the same window, and the glm5 spec route's penalized
    /// verify rows are gated against `penalized_logits` bit for bit
    /// (`gpu_device_penalties_are_bit_identical_to_the_host_sampler`). Any change to the
    /// arithmetic here is a change to the spec route's numerics too; keep the two in step.
    fn apply_penalties_dense(&self, cand: &mut [(u32, f32)]) {
        let n = self.cfg.penalty_last_n;
        if n == 0 {
            return;
        }
        if self.cfg.penalty_repeat == 1.0
            && self.cfg.penalty_freq == 0.0
            && self.cfg.penalty_present == 0.0
        {
            return;
        }
        let start = self.history.len().saturating_sub(n);
        let window = &self.history[start..];
        if window.is_empty() {
            return;
        }
        debug_assert_eq!(
            self.penalty_counts
                .values()
                .map(|&n| n as usize)
                .sum::<usize>(),
            window.len(),
            "incremental penalty counts must cover the active history window"
        );
        for (&id, &cnt) in self.penalty_counts.iter() {
            let Some(c) = cand.get_mut(id as usize) else {
                // A count for an id outside the logits row: a vocab/count mismatch, which is a
                // caller bug rather than something to silently skip. Loud in debug, inert in
                // release (dropping the penalty is strictly safer than indexing out of bounds).
                debug_assert!(
                    false,
                    "penalty count for id {id} is outside the {} candidate row",
                    cand.len()
                );
                continue;
            };
            debug_assert_eq!(
                c.0, id,
                "apply_penalties_dense requires an index-aligned candidate slice"
            );
            // repeat: llama divides if logit>0 else multiplies (penalize toward 0)
            if self.cfg.penalty_repeat != 1.0 {
                if c.1 > 0.0 {
                    c.1 /= self.cfg.penalty_repeat;
                } else {
                    c.1 *= self.cfg.penalty_repeat;
                }
            }
            c.1 -= cnt as f32 * self.cfg.penalty_freq;
            c.1 -= self.cfg.penalty_present; // presence: applied once if count>0
        }
    }

    /// The penalized logits row this sampler would score the next token from, as bytes: a
    /// copy of `logits` with the active penalty window applied through the ONE serving
    /// penalty pass (`apply_penalties_dense`). Identity when penalties are off (window
    /// empty or every coefficient neutral), exactly as `sample` sees it.
    ///
    /// EXISTS AS THE ORACLE SEAM for the device penalty kernels (lane/spec-exclusions-
    /// 20260902): a spec route that penalizes verify rows on device claims "the same bits
    /// the plain sampler produces", and that claim needs the plain sampler's bits to compare
    /// against, not a re-derivation in a test. `sample` only ever returns the chosen id.
    pub fn penalized_logits(&self, logits: &[f32]) -> Vec<f32> {
        let mut cand: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| (i as u32, l))
            .collect();
        self.apply_penalties_dense(&mut cand);
        cand.into_iter().map(|(_, l)| l).collect()
    }

    /// llama.cpp penalty: for each token in the last-n history, repeat-divide/multiply its logit
    /// and apply frequency*count + presence. (llama-sampler.cpp penalties.)
    ///
    /// THE ORACLE, not the serving path. `apply_penalties_dense` replaced it on the hot path
    /// 2026-09-01; this form is retained verbatim so the replacement has something to be proven
    /// bit-identical against, which is worth more than deleting it.
    #[cfg(test)]
    fn apply_penalties_scan_reference(&self, cand: &mut [(u32, f32)]) {
        let n = self.cfg.penalty_last_n;
        if n == 0 {
            return;
        }
        if self.cfg.penalty_repeat == 1.0
            && self.cfg.penalty_freq == 0.0
            && self.cfg.penalty_present == 0.0
        {
            return;
        }
        let start = self.history.len().saturating_sub(n);
        let window = &self.history[start..];
        if window.is_empty() {
            return;
        }
        debug_assert_eq!(
            self.penalty_counts
                .values()
                .map(|&n| n as usize)
                .sum::<usize>(),
            window.len(),
            "incremental penalty counts must cover the active history window"
        );
        for c in cand.iter_mut() {
            if let Some(&cnt) = self.penalty_counts.get(&c.0) {
                // repeat: llama divides if logit>0 else multiplies (penalize toward 0)
                if self.cfg.penalty_repeat != 1.0 {
                    if c.1 > 0.0 {
                        c.1 /= self.cfg.penalty_repeat;
                    } else {
                        c.1 *= self.cfg.penalty_repeat;
                    }
                }
                c.1 -= cnt as f32 * self.cfg.penalty_freq;
                c.1 -= self.cfg.penalty_present; // presence: applied once if count>0
            }
        }
    }
}

fn argmax_u32(logits: &[f32]) -> u32 {
    let mut best = 0u32;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i as u32;
        }
    }
    best
}

/// Stable softmax over candidate logits, writing probs back into the logit slot.
fn softmax_inplace(cand: &mut [(u32, f32)]) {
    let maxl = cand.iter().map(|c| c.1).fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for c in cand.iter_mut() {
        let e = (c.1 - maxl).exp();
        c.1 = e;
        sum += e;
    }
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for c in cand.iter_mut() {
        c.1 *= inv;
    }
}

/// SplitMix64 — deterministic seedable RNG (so a fixed seed reproduces the token stream for the
/// validation gate). Not crypto; fine for sampling.
struct SplitMix64 {
    state: u64,
}
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64 {
            state: seed.wrapping_add(0x9E3779B97F4A7C15),
        }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// uniform f32 in [0,1).
    fn next_f32(&mut self) -> f32 {
        // top 24 bits -> [0,1)
        ((self.next_u64() >> 40) as f32) / (1u32 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted_counts(s: &Sampler) -> Vec<(u32, u32)> {
        let mut counts = s.penalty_counts();
        counts.sort_unstable_by_key(|&(id, _)| id);
        counts
    }

    /// THE GATE for the 2026-09-01 host-cost fix (lane/glm5-host-audit): the dense penalty pass
    /// must reproduce the O(n_vocab) scan reference BIT FOR BIT, not merely closely.
    ///
    /// Compared as raw bit patterns (`to_bits`), because `==` on f32 would call two different
    /// NaNs equal and would call `-0.0` and `0.0` equal — and `penalty_present` subtraction can
    /// produce exactly `-0.0`. The inputs deliberately include negative logits (the repeat rule
    /// branches on sign), repeated tokens (frequency count > 1), a token never generated (must be
    /// untouched), and the three penalty coefficients both neutral and active.
    #[test]
    fn dense_penalties_match_the_scan_reference_bitwise() {
        // A small deterministic LCG: this gate must not depend on a dev-dependency.
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as u32) as f32 / (u32::MAX >> 1) as f32 - 1.0
        };

        let vocab = 257usize; // prime-ish, and > any window used below
        let mut cases = 0;
        for &(repeat, freq, present) in &[
            (1.0f32, 0.0f32, 0.0f32), // all neutral: both forms must no-op
            (1.1, 0.0, 0.0),          // repeat only (exercises the sign branch)
            (1.0, 0.5, 0.0),          // frequency only (count-scaled)
            (1.0, 0.0, 1.5),          // presence only — the q38 non-thinking vendor arm
            (1.1, 0.5, 1.5),          // all three together
            (0.8, -0.25, -0.5),       // below/negative coefficients are legal API values
        ] {
            for &window in &[1usize, 3, 64, 8192] {
                let cfg = SamplerConfig {
                    temperature: 1.0,
                    penalty_last_n: window,
                    penalty_repeat: repeat,
                    penalty_freq: freq,
                    penalty_present: present,
                    ..SamplerConfig::default()
                };
                let mut s = Sampler::new(cfg);
                // Feed a history with repeats, and never feed id 0 or id 256 so the "untouched
                // candidate" case is covered at both ends of the row.
                for step in 0..40u32 {
                    s.accept(1 + (step * 7) % 200);
                    s.accept(1 + (step * 3) % 50); // guarantees counts > 1
                }

                let base: Vec<(u32, f32)> = (0..vocab).map(|i| (i as u32, next() * 8.0)).collect();
                let mut dense = base.clone();
                let mut reference = base.clone();
                s.apply_penalties_dense(&mut dense);
                s.apply_penalties_scan_reference(&mut reference);

                assert_eq!(dense.len(), reference.len());
                for (i, (d, r)) in dense.iter().zip(reference.iter()).enumerate() {
                    assert_eq!(d.0, r.0, "candidate id moved at {i}");
                    assert_eq!(
                        d.1.to_bits(),
                        r.1.to_bits(),
                        "BIT DIVERGENCE at id {i} (repeat={repeat} freq={freq} \
                         present={present} window={window}): dense {} vs reference {}",
                        d.1,
                        r.1
                    );
                }
                // Ids never generated must be untouched — otherwise "identical" could be two
                // equally-wrong passes.
                assert_eq!(
                    dense[0].1.to_bits(),
                    base[0].1.to_bits(),
                    "id 0 was penalized"
                );
                assert_eq!(
                    dense[256].1.to_bits(),
                    base[256].1.to_bits(),
                    "id 256 was penalized"
                );
                cases += 1;
            }
        }
        assert_eq!(cases, 24, "the coefficient x window matrix must all run");
    }

    /// The precondition the dense form trades correctness for speed on: the candidate row it is
    /// handed is dense and index-aligned. Asserted on the REAL call path (`sample`), so a future
    /// caller that filters before penalizing cannot quietly pass.
    #[test]
    fn the_sampled_path_penalizes_the_token_it_generated() {
        let cfg = SamplerConfig {
            temperature: 1.0,
            penalty_last_n: 64,
            penalty_present: 100.0, // large enough that the penalized id cannot win
            seed: 7,
            ..SamplerConfig::default()
        };
        let mut s = Sampler::new(cfg);
        // Logits that make id 2 the runaway favourite, then penalize exactly id 2.
        let logits = vec![0.0, 0.0, 20.0, 0.0];
        s.accept(2);
        for _ in 0..32 {
            assert_ne!(
                s.sample(&logits),
                2,
                "presence penalty on the generated id must move the draw off it, which only \
                 happens if the dense pass indexed the right candidate"
            );
        }
    }

    #[test]
    fn greedy_is_argmax() {
        let mut s = Sampler::new(SamplerConfig::default()); // temp 0
        let logits = vec![0.1, 5.0, 2.0, -1.0];
        assert_eq!(s.sample(&logits), 1);
    }

    #[test]
    fn temp_sampling_deterministic_with_seed() {
        let cfg = SamplerConfig {
            temperature: 1.0,
            seed: 42,
            ..Default::default()
        };
        let logits = vec![1.0, 2.0, 3.0, 0.5];
        let a = Sampler::new(cfg.clone()).sample(&logits);
        let b = Sampler::new(cfg).sample(&logits);
        assert_eq!(a, b, "same seed must reproduce the draw");
        assert!(a < 4);
    }

    #[test]
    fn top_k_one_is_argmax() {
        let cfg = SamplerConfig {
            temperature: 1.0,
            top_k: 1,
            seed: 7,
            ..Default::default()
        };
        let logits = vec![0.1, 5.0, 2.0, -1.0];
        assert_eq!(
            Sampler::new(cfg).sample(&logits),
            1,
            "top_k=1 collapses to argmax"
        );
    }

    #[test]
    fn min_p_keeps_only_high_prob() {
        // logit 10 dominates; min_p 0.5 should drop the rest -> always pick id 2.
        let cfg = SamplerConfig {
            temperature: 1.0,
            min_p: 0.5,
            seed: 3,
            ..Default::default()
        };
        let logits = vec![0.0, 0.0, 10.0, 0.0];
        for _ in 0..16 {
            assert_eq!(Sampler::new(cfg.clone()).sample(&logits), 2);
        }
    }

    #[test]
    fn repeat_penalty_suppresses_recent() {
        // greedy + heavy repeat penalty: id 1 is argmax but recently emitted -> should drop it.
        let cfg = SamplerConfig {
            penalty_last_n: 8,
            penalty_repeat: 100.0,
            ..Default::default()
        };
        let mut s = Sampler::new(cfg);
        s.accept(1); // 1 was just emitted
        let logits = vec![4.0, 5.0, 4.5, 1.0]; // raw argmax = 1
        let got = s.sample(&logits);
        assert_ne!(
            got, 1,
            "recent token must be penalized out of greedy argmax"
        );
        assert_eq!(got, 2, "next-highest after penalizing 1");
    }

    #[test]
    fn penalty_counts_follow_the_sliding_window() {
        let cfg = SamplerConfig {
            penalty_last_n: 4,
            penalty_repeat: 1.1,
            penalty_freq: 0.5,
            penalty_present: 0.25,
            ..Default::default()
        };
        let mut s = Sampler::new(cfg);
        for tok in [7, 8, 7, 9] {
            s.accept(tok);
        }
        assert_eq!(sorted_counts(&s), vec![(7, 2), (8, 1), (9, 1)]);

        s.accept(8); // active window is now [8, 7, 9, 8]
        assert_eq!(sorted_counts(&s), vec![(7, 1), (8, 2), (9, 1)]);
        s.accept(10); // active window is now [7, 9, 8, 10]
        assert_eq!(sorted_counts(&s), vec![(7, 1), (8, 1), (9, 1), (10, 1)]);
        s.accept(11); // active window is now [9, 8, 10, 11]; final 7 expires
        assert_eq!(sorted_counts(&s), vec![(8, 1), (9, 1), (10, 1), (11, 1)]);
        assert!(!s.penalty_counts().iter().any(|&(id, n)| id == 7 || n == 0));
    }

    #[test]
    fn full_context_and_disabled_penalty_counts_are_exact() {
        let mut full = Sampler::new(SamplerConfig {
            penalty_last_n: usize::MAX,
            penalty_present: 1.5,
            ..Default::default()
        });
        for tok in [3, 3, 4, 5, 3] {
            full.accept(tok);
        }
        assert_eq!(sorted_counts(&full), vec![(3, 3), (4, 1), (5, 1)]);

        let mut neutral = Sampler::new(SamplerConfig {
            penalty_last_n: usize::MAX,
            ..Default::default()
        });
        neutral.accept(3);
        assert!(neutral.penalty_counts().is_empty());
    }
}

/// SESSION-RESUME SAMPLER PREDICATE teeth (lane/session-resume-sampler-predicate-20260820).
/// CPU-only, no GPU: the predicate is a pure function, so its whole contract is testable here and
/// a regression cannot hide behind "needs a card".
///
/// TEETH BOTH DIRECTIONS is the point. Every refusal test also asserts that `legacy_admits`
/// ADMITS the same pair — the pre-lane probe compared prompts and never samplers — so the test is
/// decisive (it fails on the old code) rather than tautological (passing because the pair was
/// never resumable for some other reason).
#[cfg(test)]
mod resume_sampler_predicate_tests {
    use super::*;

    /// The vendor-default sampled shape the flip makes the majority of traffic.
    fn vendor() -> SamplerConfig {
        SamplerConfig {
            temperature: 0.7,
            top_k: 20,
            top_p: 0.95,
            seed: 20260820,
            ..Default::default()
        }
    }

    /// Today's pure-temp shape — the one that parks a `graph_s`.
    fn pure_temp() -> SamplerConfig {
        SamplerConfig {
            temperature: 0.7,
            seed: 20260820,
            ..Default::default()
        }
    }

    fn id(cfg: &SamplerConfig) -> SamplerIdentity {
        SamplerIdentity::of(cfg)
    }

    // ---- direction 1: a SAME-sampler resume still resumes (no regression) ----

    #[test]
    fn identical_sampler_resumes() {
        for cfg in [pure_temp(), vendor(), SamplerConfig::default()] {
            assert_eq!(
                id(&cfg).mismatch(&id(&cfg)),
                None,
                "a request must resume a session its own sampler shaped: {cfg:?}"
            );
        }
    }

    #[test]
    fn disabled_sentinels_are_the_same_program() {
        // top_p >= 1.0, min_p <= 0.0, top_k == 0 all mean OFF; a client that spells OFF
        // differently on turn 2 must not lose its cache.
        let a = SamplerConfig {
            temperature: 0.7,
            top_p: 1.0,
            min_p: 0.0,
            ..Default::default()
        };
        let b = SamplerConfig {
            temperature: 0.7,
            top_p: 1.5,
            min_p: -1.0,
            ..Default::default()
        };
        assert_eq!(id(&a).mismatch(&id(&b)), None, "off spelled two ways");
    }

    #[test]
    fn greedy_temperature_encodings_are_one_program() {
        let a = SamplerConfig {
            temperature: 0.0,
            ..Default::default()
        };
        let b = SamplerConfig {
            temperature: -1.0,
            ..Default::default()
        };
        assert_eq!(id(&a).mismatch(&id(&b)), None, "temp<=0 is one regime");
    }

    #[test]
    fn neutral_penalty_coefficients_equal_penalties_absent() {
        // penalty_last_n set but every coefficient neutral == `pen_on == false` in spec.rs.
        let a = SamplerConfig {
            temperature: 0.7,
            penalty_last_n: 64,
            penalty_repeat: 1.0,
            penalty_freq: 0.0,
            penalty_present: 0.0,
            ..Default::default()
        };
        let b = SamplerConfig {
            temperature: 0.7,
            penalty_last_n: 0,
            ..Default::default()
        };
        assert_eq!(
            id(&a).mismatch(&id(&b)),
            None,
            "an inert penalty window is not a penalty change"
        );
    }

    // ---- direction 2: a sampler-DIFFERING resume refuses, and names the field ----

    #[test]
    fn the_reproduced_collision_pair_refuses_and_names_a_filter() {
        // The exact pair the predecessor reproduced on a live server: turn 1 pure-temp parks,
        // turn 2 adds top_p 0.95 / top_k 20 and resumes. Same seed, same temperature.
        let parked = id(&pure_temp());
        let incoming = id(&vendor());
        let field = incoming
            .mismatch(&parked)
            .expect("the reproduced collision pair must refuse");
        assert_eq!(field, "top_k", "coarsest-first order names top_k here");
        // DECISIVE: the pre-lane probe admitted exactly this pair.
        assert!(
            incoming.legacy_admits(&parked),
            "legacy must admit the collision pair, or this test proves nothing"
        );
    }

    #[test]
    fn every_compared_field_refuses_on_its_own_and_names_itself() {
        let base = pure_temp();
        // A penalized base, so the three coefficients can each move ALONE: with penalties off on
        // the parked side, turning any of them on also moves `penalty_last_n`, and coarsest-first
        // order would (correctly) name the window instead of the coefficient.
        let pen_base = SamplerConfig {
            penalty_last_n: 64,
            penalty_repeat: 1.1,
            penalty_freq: 0.5,
            penalty_present: 0.5,
            ..base.clone()
        };
        // (field, parked, incoming) — exactly one canonical field differs in each row.
        let cases: [(&str, SamplerConfig, SamplerConfig); 9] = [
            (
                "regime",
                base.clone(),
                SamplerConfig {
                    temperature: 0.0,
                    ..base.clone()
                },
            ),
            (
                "temperature",
                base.clone(),
                SamplerConfig {
                    temperature: 0.8,
                    ..base.clone()
                },
            ),
            (
                "top_k",
                base.clone(),
                SamplerConfig {
                    top_k: 20,
                    ..base.clone()
                },
            ),
            (
                "top_p",
                base.clone(),
                SamplerConfig {
                    top_p: 0.95,
                    ..base.clone()
                },
            ),
            (
                "min_p",
                base.clone(),
                SamplerConfig {
                    min_p: 0.05,
                    ..base.clone()
                },
            ),
            (
                "penalty_last_n",
                pen_base.clone(),
                SamplerConfig {
                    penalty_last_n: 128,
                    ..pen_base.clone()
                },
            ),
            (
                "penalty_repeat",
                pen_base.clone(),
                SamplerConfig {
                    penalty_repeat: 1.2,
                    ..pen_base.clone()
                },
            ),
            (
                "penalty_freq",
                pen_base.clone(),
                SamplerConfig {
                    penalty_freq: 0.6,
                    ..pen_base.clone()
                },
            ),
            (
                "penalty_present",
                pen_base.clone(),
                SamplerConfig {
                    penalty_present: 0.6,
                    ..pen_base.clone()
                },
            ),
        ];
        for (expect, parked_cfg, cfg) in cases {
            let parked = id(&parked_cfg);
            let incoming = id(&cfg);
            assert_eq!(
                incoming.mismatch(&parked),
                Some(expect),
                "changing {expect} alone must refuse and name {expect} ({cfg:?})"
            );
            assert!(
                incoming.legacy_admits(&parked),
                "legacy must admit the {expect} change, or the refusal test is tautological"
            );
        }
        // Turning penalties ON from an unpenalized parked session is a `penalty_last_n` refusal —
        // the coarsest true statement about that pair, asserted so the order is pinned.
        assert_eq!(
            id(&pen_base).mismatch(&id(&base)),
            Some("penalty_last_n"),
            "penalties on vs off is named at the window, not at a coefficient"
        );
    }

    #[test]
    fn greedy_to_sampled_and_back_both_refuse_as_regime() {
        let g = id(&SamplerConfig::default());
        let s = id(&pure_temp());
        assert_eq!(s.mismatch(&g), Some("regime"));
        assert_eq!(g.mismatch(&s), Some("regime"));
    }

    // ---- the seed decision, pinned so it cannot change silently ----

    #[test]
    fn seed_alone_does_not_refuse() {
        // DELIBERATE (see SamplerIdentity::mismatch): the draft graph is re-keyed on seed by
        // SampledGraphKey and the session's Philox counters are counter-based, so a changed seed
        // is sound — and comparing it would refuse every seed-omitting sampled conversation,
        // because an omitted seed draws fresh per-request entropy.
        let a = pure_temp();
        let b = SamplerConfig {
            seed: 999,
            ..a.clone()
        };
        assert_eq!(
            id(&a).mismatch(&id(&b)),
            None,
            "seed is carried but not compared"
        );
        assert_ne!(id(&a).seed(), id(&b).seed(), "the seed is still recorded");
    }

    #[test]
    fn mismatch_is_symmetric_and_identity_is_an_equivalence() {
        let a = id(&pure_temp());
        let b = id(&vendor());
        assert_eq!(a.mismatch(&b).is_some(), b.mismatch(&a).is_some());
        assert_eq!(a.mismatch(&a), None);
        assert_eq!(b.mismatch(&b), None);
    }
}
