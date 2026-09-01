//! Mechanism-generic speculative-decode ORACLE core (lane 10, owner directive
//! 2026-08-19): the greedy propose-then-verify arbitration loop, family-agnostic.
//!
//! This is the CPU-oracle twin of the seam memra-engine already carries for the
//! qwen3.8 drafter program (wt-dspark-q38 `crates/memra-engine/src/spec.rs`):
//! `SpecSession` (committed / next_pred / pending-carry / turn checkpoint),
//! `generate_spec*` (the draft → batched-verify → accept-walk → rollback round
//! orchestrator, with the same greedy-exactness contract stated in its header), and
//! `VerifyCkpt` (retained per-layer state-rebuild inputs → replay-free rollback).
//! Those engine structs are the EXTRACTION CANDIDATES for a shared serving core; this
//! module defines the same seam at oracle altitude so a drafter family plugs in as:
//!
//!   - [`TrunkOracle`]: prefill/step forwards that ALSO emit the family's trunk tap
//!     per position (qwen3.8: the pre-output_norm hidden of the last committed row;
//!     dsv4 DSpark: the concat of hc-mean hiddens at layers 40/41/42).
//!   - [`OracleDrafter`]: family state priming, the per-committed-position state
//!     advance (`on_commit` — dsv4: main_kv ring writes; qwen3.8: MtpScratch KV), and
//!     the proposal call (dsv4: one parallel noise-block forward + markov chaining;
//!     qwen3.8 NextN: a k-step chain of head forwards).
//!
//! The loop itself is family-blind: it owns WHEN the drafter is consulted (once per
//! round), how drafts are verified (sequential trunk steps — for greedy this is
//! mathematically identical to batched verification and to plain greedy, see
//! DSPARK-SEMANTICS §2), and the arbitration bookkeeping. Verification NEVER advances
//! trunk state with a rejected token, so the oracle loop needs no rollback — the
//! engine-side twin verifies batched and rolls back via the VerifyCkpt pattern.

/// One drafter proposal. `out_ids[0]` is the input token (the trunk's sampled next
/// token); `out_ids[1..]` are the drafted continuations. Margins/top1 are the
/// adjudication instruments (near-tie realization-flip policy, lane receipts).
pub struct Proposal {
    pub out_ids: Vec<u32>,
    pub confidence: Vec<f32>,
    pub margins: Vec<f32>,
    pub top1_logits: Vec<f32>,
}

/// Family seam: the drafter side.
pub trait OracleDrafter {
    /// Prime family state from the prefill taps `[s, tap_width]`.
    fn prime_prefill(&mut self, taps: &[f32], s: usize);
    /// Advance family state for ONE committed real position (called after EVERY
    /// trunk step — accepted positions included; rejected drafts never reach here).
    fn on_commit(&mut self, tap_row: &[f32], pos: usize);
    /// Propose drafts continuing after `input_token`, given the tap of the position
    /// the trunk just consumed (`start_pos`). Must not mutate family state.
    fn propose(&mut self, input_token: u32, tap_row: &[f32], start_pos: usize) -> Proposal;
}

/// Family seam: the trunk side. `forward` covers prefill (start_pos == 0, ids.len()
/// may be > 1) and decode (one id); returns (last-position logits, per-position taps
/// `[ids.len(), tap_width]`).
pub trait TrunkOracle {
    fn tap_width(&self) -> usize;
    fn forward(&mut self, ids: &[u32], start_pos: usize) -> (Vec<f32>, Vec<f32>);
}

/// Family seam: batched T=k+1 verification with §3.1 commit/rollback (the engine
/// shape — one trunk pass per round, then the accepted prefix's state is made
/// permanent and the rejected suffix's rolled back). The CPU-oracle contract this
/// trait gates: after `commit(n)`, every trunk cache class is bit-identical to plain
/// sequential decode of exactly the first n positions of the round.
pub trait TrunkOracleBatched: TrunkOracle {
    /// Forward ids[0..t] at positions pos0..pos0+t-1 in ONE pass, provisionally
    /// advancing state for all t. Returns (logits `[t, vocab]` — every position's
    /// row, the accept walk needs them all — and taps `[t, tap_width]`).
    fn verify_batch(&mut self, ids: &[u32], pos0: usize) -> (Vec<f32>, Vec<f32>);
    /// Make the first `n_commit` positions of the open round permanent; roll back
    /// the rest. Must be called exactly once per `verify_batch`.
    fn commit(&mut self, n_commit: usize);
}

/// One verify round's bookkeeping (mirrors the engine session's telemetry rows).
pub struct SpecRound {
    pub start_pos: usize,
    pub drafts: Vec<u32>,
    pub margins: Vec<f32>,
    pub top1: Vec<f32>,
    pub accepts: usize,
    /// draft slots actually verified (the final round may be truncated by the
    /// generation budget).
    pub verified: usize,
}

pub struct SpecRunOut {
    pub tokens: Vec<u32>,
    pub rounds: Vec<SpecRound>,
    /// trunk logits of the LAST verify step (banked by the gates).
    pub last_logits: Vec<f32>,
}

/// The greedy propose-then-verify loop (spec==plain identity law): drafts are
/// consulted once per round; each trunk decode step verifies the pending front;
/// the trunk's own argmax is ALWAYS the emitted token (bonus/correction at a
/// mismatch), so the output stream is plain greedy by construction.
///
/// `override_next(step, argmax, logits) -> token` is the gate's instrumentation
/// hook (banked-trajectory realization-flip adjudication + correction); identity
/// runs pass `|_, t, _| t`. `observe_proposal` sees every drafter call (digests).
pub fn run_spec_greedy(
    trunk: &mut dyn TrunkOracle,
    drafter: &mut dyn OracleDrafter,
    prompt: &[u32],
    n_new: usize,
    mut override_next: impl FnMut(usize, u32, &[f32]) -> u32,
    mut observe_proposal: impl FnMut(&Proposal),
) -> SpecRunOut {
    let tap_w = trunk.tap_width();
    let p0 = prompt.len();
    let (pre_logits, pre_taps) = trunk.forward(prompt, 0);
    drafter.prime_prefill(&pre_taps, p0);
    let mut t = crate::dsv4_decode::argmax(&pre_logits);
    t = override_next(usize::MAX, t, &pre_logits); // step tag MAX = the prefill emit
    let mut mh_last = pre_taps[(p0 - 1) * tap_w..p0 * tap_w].to_vec();
    let mut tokens: Vec<u32> = Vec::with_capacity(n_new);
    let mut pending: std::collections::VecDeque<u32> = Default::default();
    let mut rounds: Vec<SpecRound> = Vec::new();
    let mut cur: Option<SpecRound> = None;
    let mut last_logits: Vec<f32> = Vec::new();
    for step in 0..n_new {
        let m = p0 + step; // t sits at index m
        if pending.is_empty() {
            if let Some(r) = cur.take() {
                rounds.push(r);
            }
            let prop = drafter.propose(t, &mh_last, m - 1);
            observe_proposal(&prop);
            pending = prop.out_ids[1..].iter().cloned().collect();
            cur = Some(SpecRound {
                start_pos: m - 1,
                drafts: prop.out_ids[1..].to_vec(),
                margins: prop.margins,
                top1: prop.top1_logits,
                accepts: 0,
                verified: 0,
            });
        }
        tokens.push(t);
        if step + 1 == n_new {
            break;
        }
        let (logits, taps) = trunk.forward(&[t], m);
        drafter.on_commit(&taps, m);
        let mut t_next = crate::dsv4_decode::argmax(&logits);
        t_next = override_next(step, t_next, &logits);
        let d = pending.pop_front().expect("pending nonempty");
        let round = cur.as_mut().expect("open round");
        round.verified += 1;
        if d == t_next {
            round.accepts += 1;
        } else {
            pending.clear();
        }
        mh_last = taps;
        last_logits = logits;
        t = t_next;
    }
    if let Some(r) = cur.take() {
        rounds.push(r);
    }
    SpecRunOut {
        tokens,
        rounds,
        last_logits,
    }
}

/// The BATCHED greedy propose-then-verify loop (§3.1, iteration 3): one drafter call
/// and ONE batched trunk pass per round (T = 1 + verifiable drafts), then the accept
/// walk, `commit(accepted+1)`, and per-committed-position drafter state advance —
/// main_kv rings advance for EVERY accepted position (DSPARK-SEMANTICS §2), never
/// for a rejected one. For greedy decoding this loop is mathematically identical to
/// [`run_spec_greedy`] (logits at a position depend only on inputs ≤ it), and the
/// round/budget accounting below reproduces the sequential loop's exactly (including
/// the budget-truncated final round and its pending-carry no-propose tail), so the
/// two loops' proposal streams and output tokens are comparable item-for-item.
///
/// `on_round(trunk, drafter, &round)` fires after each round's commit — the §3.1
/// gate digests cache state there and compares it against a sequential twin.
pub fn run_spec_greedy_batched<T, D>(
    trunk: &mut T,
    drafter: &mut D,
    prompt: &[u32],
    n_new: usize,
    mut override_next: impl FnMut(usize, u32, &[f32]) -> u32,
    mut observe_proposal: impl FnMut(&Proposal),
    mut on_round: impl FnMut(&mut T, &mut D, &SpecRound),
) -> SpecRunOut
where
    T: TrunkOracleBatched + ?Sized,
    D: OracleDrafter + ?Sized,
{
    let tap_w = trunk.tap_width();
    let p0 = prompt.len();
    let (pre_logits, pre_taps) = trunk.forward(prompt, 0);
    drafter.prime_prefill(&pre_taps, p0);
    let mut t = crate::dsv4_decode::argmax(&pre_logits);
    t = override_next(usize::MAX, t, &pre_logits);
    let mut mh_last = pre_taps[(p0 - 1) * tap_w..p0 * tap_w].to_vec();
    let mut tokens: Vec<u32> = Vec::with_capacity(n_new);
    let mut rounds: Vec<SpecRound> = Vec::new();
    let mut last_logits: Vec<f32> = Vec::new();
    // pending-carry parity with the sequential loop: a budget-truncated round whose
    // verified prefix was fully accepted leaves drafts pending — the final token is
    // then emitted WITHOUT a fresh proposal, exactly like the sequential loop.
    let mut carry_pending = false;
    while tokens.len() < n_new {
        if carry_pending {
            tokens.push(t);
            assert_eq!(tokens.len(), n_new, "pending carry only at the budget tail");
            break;
        }
        let m0 = p0 + tokens.len(); // t sits at index m0
        let prop = drafter.propose(t, &mh_last, m0 - 1);
        observe_proposal(&prop);
        let k = prop.out_ids.len() - 1;
        tokens.push(t);
        if tokens.len() == n_new {
            // sequential parity: the last step proposes (pending was empty), pushes
            // t and breaks before any forward — a verified-0 round.
            rounds.push(SpecRound {
                start_pos: m0 - 1,
                drafts: prop.out_ids[1..].to_vec(),
                margins: prop.margins,
                top1: prop.top1_logits,
                accepts: 0,
                verified: 0,
            });
            break;
        }
        // forwards the sequential loop would still run: positions m0..p0+n_new-2,
        // i.e. n_new - len of them — the batch never forwards a position the
        // sequential loop would not (state-parity requirement of the §3.1 gate).
        let forwards_left = n_new - tokens.len();
        let t_batch = (k + 1).min(forwards_left);
        let kv = t_batch - 1; // drafts verifiable this round
        let mut batch_ids = Vec::with_capacity(t_batch);
        batch_ids.push(t);
        batch_ids.extend_from_slice(&prop.out_ids[1..1 + kv]);
        let (logits_all, taps) = trunk.verify_batch(&batch_ids, m0);
        let vocab = logits_all.len() / t_batch;
        // accept walk: row i (position m0+i) arbitrates draft i+1; the override hook
        // sees the SAME (step, argmax, logits) stream the sequential loop produced.
        let mut c_d = 0usize; // accepted drafts
        let mut t_next = 0u32;
        for i in 0..t_batch {
            let row = &logits_all[i * vocab..(i + 1) * vocab];
            let step = tokens.len() - 1 + i; // sequential step index of this forward
            let mut a = crate::dsv4_decode::argmax(row);
            a = override_next(step, a, row);
            if i < kv && a == batch_ids[i + 1] {
                c_d += 1;
                continue;
            }
            t_next = a;
            last_logits = row.to_vec();
            break;
        }
        let n_commit = c_d + 1; // t plus the accepted drafts
        trunk.commit(n_commit);
        for i in 0..n_commit {
            drafter.on_commit(&taps[i * tap_w..(i + 1) * tap_w], m0 + i);
        }
        mh_last = taps[c_d * tap_w..(c_d + 1) * tap_w].to_vec();
        // emit the accepted drafts; t_next is the bonus/correction for next round
        for i in 0..c_d {
            tokens.push(batch_ids[i + 1]);
        }
        carry_pending = c_d == kv && kv < k;
        let round = SpecRound {
            start_pos: m0 - 1,
            drafts: prop.out_ids[1..].to_vec(),
            margins: prop.margins,
            top1: prop.top1_logits,
            accepts: c_d,
            verified: (c_d + 1).min(kv),
        };
        on_round(trunk, drafter, &round);
        rounds.push(round);
        t = t_next;
    }
    SpecRunOut {
        tokens,
        rounds,
        last_logits,
    }
}
