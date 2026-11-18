//! Dense forward pass (Stage-1, all f32, prefill of T tokens, batch=1).
//! Matches llama.cpp qwen3 graph: embed → per layer {RMSNorm, QKV, QK-norm, RoPE, SDPA, O,
//! residual, RMSNorm, SwiGLU, residual} → output_norm → lm_head.
//!
//! Activation layout: x is [n_embd, T] but we store it row-major-per-token as [T, n_embd]
//! (token t at offset t*n_embd) so cuBLASLt linear (m=T tokens, in=n_embd) works directly.

use crate::Engine;
use crate::model::Model;

impl Model {
    /// Run prefill over `tokens`, return logits [T, n_vocab] (host f32). positions = 0..T.
    pub fn forward(
        &self,
        e: &Engine,
        tokens: &[u32],
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let n_head = cfg.n_head as usize;
        let n_head_kv = cfg.n_head_kv as usize;
        let head_dim = cfg.head_dim_k as usize;
        let t = tokens.len();
        let eps = cfg.rms_eps;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // positions 0..T
        let pos: Vec<i32> = (0..t as i32).collect();
        let pos_d = e.htod_i32(&pos)?;

        // x: [T, n_embd] (token-major)
        let mut x = self.embed_tokens(e, tokens)?;
        // Fixed MoE cache-slot size (0 for a non-MoE dense model). Computed once for the whole run.
        let max_block = self.max_moe_block();

        for (il, layer) in self.layers.iter().enumerate() {
            // --- attention block ---
            let mut h = e.zeros(t * n_embd)?;
            e.rms_norm(&x, layer.attn_norm.float_data(), &mut h, n_embd, t, eps)?;

            // QKV projections: q[T, n_head*head_dim], k/v[T, n_head_kv*head_dim]
            let q_out = layer.wq.out_features(); // n_head*head_dim
            let k_out = layer.wk.out_features();
            let mut q = e.matmul(&layer.wq, &h, t)?;
            let mut k = e.matmul(&layer.wk, &h, t)?;
            let v = e.matmul(&layer.wv, &h, t)?;

            // QK-norm: RMSNorm over head_dim, per (token, head). Layout [head_dim, n_head, T]
            // == row-major rows of length head_dim. q currently [T, n_head*head_dim] which is
            // exactly n_head*T rows of head_dim if we treat each head slice as a row. The memory
            // for token t is [head0(head_dim) head1(head_dim) ...]; so rows of head_dim are
            // contiguous and number n_head*T. RMSNorm with ncols=head_dim, nrows=n_head*T works.
            // rms_norm(ncols=head_dim, nrows=n_head*T) multiplies each row by q_norm[head_dim] —
            // exactly per-head QK-norm. Rows of head_dim are contiguous in the token-major buffer.
            if let Some(qn) = &layer.q_norm {
                let mut qn_out = e.zeros(t * q_out)?;
                e.rms_norm(&q, qn.float_data(), &mut qn_out, head_dim, n_head * t, eps)?;
                q = qn_out;
            }
            if let Some(kn) = &layer.k_norm {
                let mut kn_out = e.zeros(t * k_out)?;
                e.rms_norm(
                    &k,
                    kn.float_data(),
                    &mut kn_out,
                    head_dim,
                    n_head_kv * t,
                    eps,
                )?;
                k = kn_out;
            }

            // RoPE on q,k. Layout per token is [head_dim, n_head] contiguous. Our buffer is
            // [T, n_head*head_dim]; rope_neox expects [head_dim, n_heads, n_tokens] with grid
            // n_heads*n_tokens and head index = blockIdx % n_heads. Token-major works if we pass
            // n_heads and the kernel treats hr = token*n_heads + head. Our layout: token t at
            // t*(n_head*head_dim), head h at +h*head_dim. So hr index = t*n_head + h → matches
            // kernel's head=hr%n_heads, tok=hr/n_heads ONLY if hr = tok*n_head+head. Good.
            e.rope_neox(
                &mut q,
                &pos_d,
                head_dim,
                cfg.rope_dim_count as usize,
                n_head,
                t,
                cfg.rope_freq_base,
                1.0,
            )?;
            e.rope_neox(
                &mut k,
                &pos_d,
                head_dim,
                cfg.rope_dim_count as usize,
                n_head_kv,
                t,
                cfg.rope_freq_base,
                1.0,
            )?;

            // SDPA: q[head_dim,n_head,T], k/v[head_dim,n_head_kv,T] (T_kv = T for prefill).
            // Our buffers are token-major [T, heads*head_dim] == [head_dim, heads, T] interpreting
            // index (d, head, tok) at tok*(heads*head_dim)+head*head_dim+d. The SDPA kernel indexes
            // Q at (qt*n_head+head)*head_dim+d — identical. Good.
            let mut attn = e.zeros(t * q_out)?;
            e.sdpa_naive(
                &q, &k, &v, &mut attn, head_dim, n_head, n_head_kv, t, t, scale, true,
            )?;

            // O projection: attn[T, n_head*head_dim] @ wo[n_embd, n_head*head_dim]^T
            let o = e.matmul(&layer.wo, &attn, t)?;

            // residual 1
            let mut x1 = e.zeros(t * n_embd)?;
            e.add(&x, &o, &mut x1, t * n_embd)?;

            // --- ffn block: dense SwiGLU or routed MoE (OLMoE) ---
            let mut z = e.zeros(t * n_embd)?;
            e.rms_norm(&x1, layer.ffn_norm.float_data(), &mut z, n_embd, t, eps)?;
            let down = match &layer.ffn {
                crate::hybrid::Ffn::Dense {
                    ffn_gate,
                    ffn_up,
                    ffn_down,
                    ffn_down_pqs,
                } => {
                    let n_ff = ffn_gate.out_features();
                    let gate = e.matmul(ffn_gate, &z, t)?;
                    let up = e.matmul(ffn_up, &z, t)?;
                    let mut act = e.zeros(t * n_ff)?;
                    crate::hybrid::HybridModel::ffn_act(e, cfg, &gate, &up, &mut act, t * n_ff)?;
                    // AWQ (memra#253): the down projection's input must carry the per-input-channel
                    // scale when the artifact was calibrated; None leaves the buffer untouched.
                    let __pqs =
                        e.pre_quant_scaled(&act, ffn_down_pqs.as_ref(), ffn_down.in_features(), t)?;
                    e.matmul(ffn_down, __pqs.as_ref().unwrap_or(&act), t)?
                }
                crate::hybrid::Ffn::Moe(m) => {
                    crate::hybrid::HybridModel::moe_ffn(e, m, &z, t, cfg, il as u16, max_block)?
                }
            };

            // residual 2
            let mut x2 = e.zeros(t * n_embd)?;
            e.add(&x1, &down, &mut x2, t * n_embd)?;
            x = x2;
        }

        // final norm + lm_head
        let mut hn = e.zeros(t * n_embd)?;
        e.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, t, eps)?;
        let logits = e.matmul(&self.output, &hn, t)?;
        let host = e.dtoh(&logits)?;
        Ok(host)
    }

    /// Logits for just the last token (the decode-relevant row).
    pub fn forward_last(
        &self,
        e: &Engine,
        tokens: &[u32],
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let all = self.forward(e, tokens)?;
        let n_vocab = self.output.out_features();
        let t = tokens.len();
        Ok(all[(t - 1) * n_vocab..t * n_vocab].to_vec())
    }
}

/// argmax helper: the first index of the largest value, NaN skipped. A row with no candidate
/// (every value NaN or -inf, or no value) has no argmax and gives `TOKEN_NONE`, the id the
/// device argmax writes for it, so `check_token` refuses both the same way.
pub fn argmax(logits: &[f32]) -> usize {
    let mut best = crate::TOKEN_NONE as usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best
}

/// top-1/top-2 (id, value) pairs — the greedy near-tie margin is v1 - v2.
pub fn top2(logits: &[f32]) -> (usize, f32, usize, f32) {
    let (mut i1, mut v1, mut i2, mut v2) = (0usize, f32::NEG_INFINITY, 0usize, f32::NEG_INFINITY);
    for (i, &v) in logits.iter().enumerate() {
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

/// Gate #46 verdict: BATCHED-PRIME last-position logits vs the TOKENWISE-PRIME reference.
///
/// The two primes are different numeric configs by design (prefill GEMM m=T vs decode
/// GEMV m=1 — cross-config drift class, like forward_last vs decode). An argmax flip on a
/// near-tie under small logit drift is within that law; a flip on a WIDE margin, or drift
/// beyond the calibrated ceiling, cannot come from the FP-composition class and fails hard.
///
/// Bounds (env-overridable; calibrated on the 2026-08-02 supported-set sweep,
/// research/prime-gate-coverage-20260802 — recalibrate when the kernels under them move,
/// the H100 stale-verdict law):
///   MEMRA_PRIME_GATE_MAXDIFF — full-vocab logit maxdiff ceiling (default 8.0). Measured
///     legal cross-config drift across the 144-prompt supported-set sweep: dense Q8_0 up
///     to ~1.0, MoE IQ4_XS/Q4_K_M up to 3.1, gemma QAT Q4_0 up to 5.5 (its logit scale is
///     larger); run-gen's own accepted forward-vs-decode drift is 1.39 on the q35 probe.
///     A real defect (indexing/wrong-weights) lands decades above this.
///   MEMRA_PRIME_GATE_MARGIN — tokenwise top1-top2 margin above which an argmax flip is
///     treated as structured (default 1.0). All 10 measured legal first-token flips sat
///     at margins <= 0.70 (per-position flips <= 0.92).
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum PrimeGateClass {
    Match,
    NearTieFlip,
    Structured,
}

pub struct PrimeGateVerdict {
    pub tw_argmax: usize,
    pub bp_argmax: usize,
    pub tw_margin: f32,
    pub bp_margin: f32,
    pub maxdiff: f32,
    pub class: PrimeGateClass,
}

fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

pub fn prime_gate_verdict(tokenwise: &[f32], batched: &[f32]) -> PrimeGateVerdict {
    let (t1, tv1, _, tv2) = top2(tokenwise);
    let (b1, bv1, _, bv2) = top2(batched);
    let maxdiff = tokenwise
        .iter()
        .zip(batched)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let maxdiff_bound = env_f32("MEMRA_PRIME_GATE_MAXDIFF", 8.0);
    let margin_bound = env_f32("MEMRA_PRIME_GATE_MARGIN", 1.0);
    let tw_margin = tv1 - tv2;
    let class = if !maxdiff.is_finite() || maxdiff > maxdiff_bound {
        PrimeGateClass::Structured
    } else if t1 == b1 {
        PrimeGateClass::Match
    } else if tw_margin <= margin_bound {
        PrimeGateClass::NearTieFlip
    } else {
        PrimeGateClass::Structured
    };
    PrimeGateVerdict {
        tw_argmax: t1,
        bp_argmax: b1,
        tw_margin,
        bp_margin: bv1 - bv2,
        maxdiff,
        class,
    }
}

/// The verdict of run-gen's first gate (memra#1295): the batched prefill (`forward_last`)
/// against the tokenwise decode loop at the last position.
pub struct PrefillGateVerdict {
    pub prefill_argmax: usize,
    pub decode_argmax: usize,
    /// How far the prefill holds its token over the decode's: `p[i] - p[j]`. 0 on a match.
    pub prefill_margin: f32,
    /// How far the decode holds its token over the prefill's: `d[j] - d[i]`. 0 on a match.
    pub decode_margin: f32,
    /// The larger of the two programs' differences at those two ids.
    pub spread: f32,
    /// Full-vocab maxdiff; NaN when a row holds a NaN, when both rows hold the same infinity
    /// at one id, or when the rows differ in length.
    pub maxdiff: f32,
    pub class: PrimeGateClass,
}

/// Judge a last-position row pair of run-gen's first gate. The two programs are different
/// numeric configs, like the two primes of gate #46 above, so a flip is judged on the same
/// calibrated bounds instead of asserted on:
///
/// - `Match`: the two argmax ids agree and are real ids. Passes whatever the maxdiff: this
///   gate never bounded the drift of an agreeing row, and the ceiling was calibrated on the
///   two primes.
/// - `NearTieFlip`: the ids differ, each program holds its own token over the other's by at
///   most `MEMRA_PRIME_GATE_MARGIN`, and the maxdiff is within `MEMRA_PRIME_GATE_MAXDIFF`.
///   Reported; generation goes on.
/// - `Structured`: any other flip, including one where a row holds a NaN; and any pair where
///   a row has no finite value, so no argmax (two such rows "agree" on `TOKEN_NONE`). Fails
///   the run.
///
/// The margins are taken between the two contending ids in BOTH programs, which is stricter
/// than gate #46's tokenwise top-2 margin: a prefill that holds its token by 7 logits over a
/// decode that has the two tied is two programs 7 logits apart on that pair, not a near tie.
///
/// The gate used to label a flip by comparing the spread at the two ids with the smaller
/// margin, and then panicked on either label. That comparison cannot separate the classes:
/// with prefill picking i and decode picking j, the differences (p_i - d_i) and (d_j - p_j)
/// sum to the two margins' sum, so the larger one is at least the smaller margin for every
/// flip.
pub fn prefill_gate_verdict(prefill: &[f32], decode: &[f32]) -> PrefillGateVerdict {
    prefill_gate_verdict_with(
        prefill,
        decode,
        env_f32("MEMRA_PRIME_GATE_MAXDIFF", 8.0),
        env_f32("MEMRA_PRIME_GATE_MARGIN", 1.0),
    )
}

/// [`prefill_gate_verdict`] on explicit bounds (its unit tests read no environment).
pub fn prefill_gate_verdict_with(
    prefill: &[f32],
    decode: &[f32],
    maxdiff_bound: f32,
    margin_bound: f32,
) -> PrefillGateVerdict {
    let (i, j) = (argmax(prefill), argmax(decode));
    let maxdiff = if prefill.len() == decode.len() {
        prefill
            .iter()
            .zip(decode)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, |m, d| {
                if m.is_nan() || d.is_nan() {
                    f32::NAN
                } else {
                    m.max(d)
                }
            })
    } else {
        f32::NAN
    };
    let mut v = PrefillGateVerdict {
        prefill_argmax: i,
        decode_argmax: j,
        prefill_margin: 0.0,
        decode_margin: 0.0,
        spread: f32::NAN,
        maxdiff,
        class: PrimeGateClass::Match,
    };
    let seated = i < prefill.len() && i < decode.len() && j < prefill.len() && j < decode.len();
    if seated {
        v.spread = (prefill[i] - decode[i])
            .abs()
            .max((prefill[j] - decode[j]).abs());
    }
    // `argmax` gives `TOKEN_NONE` for a row with no finite value: not seated, and neither a
    // match nor a near tie, whatever the other row holds.
    if !seated {
        v.prefill_margin = f32::NAN;
        v.decode_margin = f32::NAN;
        v.class = PrimeGateClass::Structured;
        return v;
    }
    if i == j {
        return v;
    }
    v.prefill_margin = prefill[i] - prefill[j];
    v.decode_margin = decode[j] - decode[i];
    // Plain comparisons, so that a NaN margin or maxdiff fails its own.
    let near_tie = maxdiff <= maxdiff_bound
        && v.prefill_margin <= margin_bound
        && v.decode_margin <= margin_bound;
    v.class = if near_tie {
        PrimeGateClass::NearTieFlip
    } else {
        PrimeGateClass::Structured
    };
    v
}

#[cfg(test)]
mod prefill_gate_tests {
    use super::{PrimeGateClass, prefill_gate_verdict_with};

    /// A flat row at -4.0 with the given (id, logit) entries set.
    fn row(n: usize, set: &[(usize, f32)]) -> Vec<f32> {
        let mut l = vec![-4.0f32; n];
        for &(i, v) in set {
            l[i] = v;
        }
        l
    }

    fn class(prefill: &[f32], decode: &[f32]) -> PrimeGateClass {
        prefill_gate_verdict_with(prefill, decode, 8.0, 1.0).class
    }

    /// memra#1295's 4,100-token row: the programs hold their tokens by 0.038 and 0.117.
    #[test]
    fn near_tie_flip_is_reported_not_fatal() {
        let prefill = row(400, &[(343, 18.9964), (16, 18.9582)]);
        let decode = row(400, &[(343, 18.9904), (16, 19.1077)]);
        let v = prefill_gate_verdict_with(&prefill, &decode, 8.0, 1.0);
        assert_eq!((v.prefill_argmax, v.decode_argmax), (343, 16));
        assert!(
            (v.prefill_margin - 0.0382).abs() < 1e-3,
            "{}",
            v.prefill_margin
        );
        assert!(
            (v.decode_margin - 0.1173).abs() < 1e-3,
            "{}",
            v.decode_margin
        );
        assert!((v.spread - 0.1495).abs() < 1e-3, "{}", v.spread);
        assert_eq!(v.class, PrimeGateClass::NearTieFlip);
    }

    /// Both programs hold their token by 5 logits or more. The old inline label called this
    /// row a near tie too (spread 6 against a smaller margin of 5).
    #[test]
    fn wide_margin_flip_fails() {
        let prefill = row(64, &[(3, 10.0), (7, 4.0)]);
        let decode = row(64, &[(3, 5.0), (7, 10.0)]);
        assert_eq!(class(&prefill, &decode), PrimeGateClass::Structured);
    }

    /// One program near-tied is not enough, whichever program it is.
    #[test]
    fn flip_wide_in_one_program_fails() {
        // decode has the two ids 0.1 apart, prefill holds its token by 4
        let prefill = row(64, &[(3, 10.0), (7, 6.0)]);
        let decode = row(64, &[(3, 9.9), (7, 10.0)]);
        assert_eq!(class(&prefill, &decode), PrimeGateClass::Structured);
        // prefill has the two ids 0.1 apart, decode holds its token by 4
        let prefill = row(64, &[(3, 10.0), (7, 9.9)]);
        let decode = row(64, &[(3, 6.0), (7, 10.0)]);
        assert_eq!(class(&prefill, &decode), PrimeGateClass::Structured);
    }

    /// The margin is the one between the two contending ids, not each row's top-2 margin:
    /// both rows have a runner-up within 0.5, and the prefill holds its token over the
    /// decode's by 7. The maxdiff (7) is inside its bound, so the margin alone decides.
    #[test]
    fn flip_with_a_close_third_id_fails() {
        let prefill = row(64, &[(3, 10.0), (5, 9.9), (7, 3.0)]);
        let decode = row(64, &[(3, 9.5), (5, 9.4), (7, 10.0)]);
        let v = prefill_gate_verdict_with(&prefill, &decode, 8.0, 1.0);
        assert!(v.maxdiff <= 8.0, "{}", v.maxdiff);
        assert_eq!((v.prefill_margin, v.decode_margin), (7.0, 0.5));
        assert_eq!(v.class, PrimeGateClass::Structured);
    }

    /// The bounds are inclusive, and each one decides on its own.
    #[test]
    fn bounds_are_inclusive() {
        let prefill = row(64, &[(3, 10.0), (7, 9.0)]);
        let decode = row(64, &[(3, 9.0), (7, 10.0)]);
        let at =
            |maxdiff, margin| prefill_gate_verdict_with(&prefill, &decode, maxdiff, margin).class;
        assert_eq!(at(8.0, 1.0), PrimeGateClass::NearTieFlip);
        assert_eq!(at(8.0, 0.99), PrimeGateClass::Structured);
        assert_eq!(at(1.0, 1.0), PrimeGateClass::NearTieFlip);
        assert_eq!(at(0.99, 1.0), PrimeGateClass::Structured);
    }

    /// A flip on small margins under drift past the ceiling is not the near-tie class.
    #[test]
    fn flip_under_out_of_bounds_drift_fails() {
        let mut prefill = row(64, &[(3, 10.0), (7, 9.9)]);
        let decode = row(64, &[(3, 9.9), (7, 10.0)]);
        prefill[40] = 8.5; // 12.5 logits away from decode's -4.0
        assert_eq!(class(&prefill, &decode), PrimeGateClass::Structured);
    }

    /// A NaN anywhere in a flipped row pair, or a row with no finite value, fails.
    #[test]
    fn flip_on_a_broken_row_fails() {
        let mut prefill = row(64, &[(3, 10.0), (7, 9.9)]);
        let decode = row(64, &[(3, 9.9), (7, 10.0)]);
        prefill[40] = f32::NAN;
        let v = prefill_gate_verdict_with(&prefill, &decode, 8.0, 1.0);
        assert!(v.maxdiff.is_nan());
        assert_eq!(v.class, PrimeGateClass::Structured);
        let dead = vec![f32::NAN; 64];
        let v = prefill_gate_verdict_with(&dead, &decode, 8.0, 1.0);
        assert_eq!(v.prefill_argmax, crate::TOKEN_NONE as usize);
        assert_eq!(v.class, PrimeGateClass::Structured);
        assert_eq!(class(&decode, &dead), PrimeGateClass::Structured);
        // two dead rows "agree" on TOKEN_NONE: not a match
        assert_eq!(class(&dead, &dead), PrimeGateClass::Structured);
        assert_eq!(class(&prefill[..32], &decode), PrimeGateClass::Structured);
    }

    /// An agreeing row passes whatever its drift, as before the change.
    #[test]
    fn agreeing_row_matches_whatever_the_drift() {
        let mut prefill = row(64, &[(3, 30.0)]);
        let decode = row(64, &[(3, 30.5)]);
        prefill[40] = 8.5;
        let v = prefill_gate_verdict_with(&prefill, &decode, 8.0, 1.0);
        assert!(v.maxdiff > 8.0);
        assert_eq!((v.prefill_margin, v.decode_margin), (0.0, 0.0));
        assert_eq!(v.class, PrimeGateClass::Match);
    }
}
