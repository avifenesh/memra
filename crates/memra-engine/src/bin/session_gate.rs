//! SESSION EXACTNESS GATE: multi-turn generate_spec_session must produce, at every turn, the
//! IDENTICAL tokens plain greedy decode produces when primed with the session's full committed
//! history. Verify guarantees per-turn exactness; this gate pins the turn BOUNDARY (the resume a
//! new turn takes, its suffix prime and the cross-turn draft-KV/pairing state).
//!
//! What changed with memra#1146, where the gate failed turn 2 at K=3 with 48 tokens per turn on
//! Qwen3.8-27B (GB10 and RTX 5090) and passed at its defaults:
//!
//! - **The oracle is run-spec's** (`HybridModel::generate_spec_oracle`), not the eager
//!   `generate`. The qwen35 (GatedDeltaNet) family and Step35 serve B=1 on the batched program
//!   and their verify forward is aligned to it; the eager class is about 1e-2 away at the top
//!   logits on the fast path, and it moved turn 2's second token on a 0.02 margin. That was the
//!   failure.
//! - **A step-identity check runs first** (`HybridModel::served_step_identity`, also in
//!   run-spec): K+1 plain steps of the program the server decodes the model with against one
//!   (K+1)-row verify forward over the same tokens, two rounds, every logit bit-compared, from
//!   one fixed prompt. A verify forward in another numeric class differs in its bits on every
//!   prompt, so this does not need a near tie to see it.
//! - **A turn takes the resume the served engine takes.** An attention-only trunk resumes at its
//!   parked end. A recurrent (GDN) trunk does not: the rows past the prompt were written by
//!   verify and decode, off the prime grid, and a suffix primed from there starts from state a
//!   cold prime never computes (the grid law, memra#1091 and memra#1093, the worker's
//!   `spec_pool_grid_refusal`). On a recurrent trunk the gate refuses the parked-end resume and
//!   rewinds to the session's turn checkpoint when it sits on the grid and the prompt extends
//!   it by the prime floor, else primes cold in a fresh session. Each prime stops at the last
//!   grid boundary that leaves the prime floor behind it and captures the turn checkpoint
//!   there, as serving's stable boundary does. At least one turn must rewind, or the gate
//!   fails: a run where every turn primed cold tested no boundary.
//! - **The continuation burst** (an empty suffix, the serve pattern) is compared with the
//!   continuation of the previous turn's oracle, not with a cold prime of rows the session's
//!   own decode wrote: on a recurrent trunk those are two programs for the same rows.
//! - **A continuation burst that runs no round** (memra#1196). The server ends a burst at its
//!   entry when a request waits for admission: the burst's first commit hook says stop. A
//!   session that entered such a burst on a carried pending token must leave it pending and
//!   leave `committed` what it was: the cache rows, and the true history less the pending
//!   token. The next burst must carry on with the oracle's tokens. Before the fix the burst
//!   pushed the carried token into `committed` although no round had committed its row. Under
//!   `MEMRA_SPEC_REPLAY=1` no burst ends on a pending token and the check fails as untested.
//! - **A continuation burst the hook stops mid-burst** (memra#1207). The server ends a burst at
//!   the round that completes a stop string and parks the session for the next turn. The hook
//!   says stop after its second slice: the burst must end there with `committed` equal to the
//!   cache rows and to the true history less the pending token, and the next burst must carry
//!   on with the oracle's tokens.
//! - **A sampled burst carries its pending token** (memra#1186). Three sessions serve one
//!   seeded, frequency-penalised request: the first in one burst (no burst boundary inside the
//!   answer, the reference), the second one burst after another, the third with a no-round
//!   burst after every burst. Every burst of the second and third must end on a carried
//!   token, `committed` must stay the cache rows, and the three must emit the same tokens:
//!   the draw is keyed by position, so where a burst ends and whether a burst ran a round may
//!   not move it. A carry that drew the token after a boundary at another event, or in
//!   another penalty window, differs from the one-burst session there.
//!
//! - **A lend above the checkpoint** (`MEMRA_GATE_LEND_ABOVE=1`, memra#1221 review). Before each
//!   rewind the gate lends every trunk layer's rows `[0, committed)`, as a prefix entry whose
//!   boundary capture sat above the turn checkpoint reads them, and holds the lends through the
//!   turn. The rewind must keep those rows (the session moves to a fresh cache, its old planes to
//!   the lends) and the turn must still emit the oracle's tokens: the lent rows' bytes after the
//!   turn must be the bytes before it.
//!
//! usage: session-gate <model.gguf>   (env: MEMRA_SPEC_K, MEMRA_NGEN per turn, MEMRA_GATE_LEND_ABOVE)
use memra_engine::Engine;
use memra_engine::hybrid::HybridModel;
use memra_engine::hybrid_forward::PRIME_MIN_T;
use memra_engine::memra_gguf::GgufFile;
use memra_engine::spec::{SpecSampling, SpecSession};

/// The last grid boundary at or below `len` that leaves at least the prime floor of rows after
/// it (the worker's `grid_align_boundary_within`), or `None` when none lies past `start`.
fn stable_boundary(start: usize, len: usize, grid: usize) -> Option<usize> {
    let mut b = len / grid * grid;
    while b >= grid && len - b < PRIME_MIN_T {
        b -= grid;
    }
    (b > start).then_some(b)
}

/// Why a recurrent session cannot rewind to its turn checkpoint for a prompt of `prompt_len`
/// tokens that reproduces its committed rows (the worker's `spec_pool_grid_refusal`), or `None`
/// when it rewinds.
fn rewind_refusal(sess: &SpecSession, prompt_len: usize, grid: usize) -> Option<String> {
    let Some(pos) = sess.rewind_pos().filter(|&p| p > 0) else {
        return Some("no turn checkpoint".into());
    };
    if !pos.is_multiple_of(grid) {
        return Some(format!(
            "the checkpoint {pos} is off the {grid}-token prime grid"
        ));
    }
    if !sess.rewind_is_resident() {
        return Some(format!("the ring lapped the checkpoint {pos}"));
    }
    if prompt_len < pos + PRIME_MIN_T {
        return Some(format!(
            "the prompt extends the checkpoint {pos} by fewer than the prime floor"
        ));
    }
    None
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: session-gate <model.gguf>");
    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let model = HybridModel::load(&e, &g)?;
    let k: usize = std::env::var("MEMRA_SPEC_K")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let n_new: usize = std::env::var("MEMRA_NGEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(24);
    let grid = Engine::gdn_chunk_size();

    // Three turns of real-ish token suffixes (numeric ids are fine — the gate is exactness,
    // not acceptance).
    let turn1: Vec<u32> = (101..229).collect();
    let turn2: Vec<u32> = (350..415).collect();
    let turn3: Vec<u32> = (517..551).collect();

    let mut sess = model.new_session(&e, 4096)?;
    let recurrent = sess.cache_ref().recur.iter().any(Option::is_some);
    println!(
        "trunk: {}; turn resume: {}",
        if recurrent {
            "recurrent"
        } else {
            "attention-only"
        },
        if recurrent {
            format!(
                "the served one (rewind to an on-grid turn checkpoint, else cold; grid {grid}); \
                 the parked-end resume is refused"
            )
        } else {
            "the parked end".to_string()
        }
    );
    let mut ok = true;
    // A prompt off the prime grid with a tail above the prime floor (193 = 6 x 32 + 1 would
    // decode its tail token by token; 211 leaves 19).
    let identity_prompt: Vec<u32> = (101..229).chain(350..433).collect();
    match model.served_step_identity(&e, &identity_prompt, k + 1, 2)? {
        Some(identity) => {
            println!("{}", identity.line());
            if !identity.matches() {
                if model.step_identity_enforced() {
                    ok = false;
                } else {
                    println!(
                        "step identity is reported, not enforced, on this family: its verify \
                         forward is not declared in the serving class (memra#1157)"
                    );
                }
            }
        }
        None => println!("STEP-IDENTITY: n/a (no MTP head or no generic verify forward)"),
    }
    let mut rewound_turns = 0usize;
    let lend_above = std::env::var("MEMRA_GATE_LEND_ABOVE").is_ok_and(|v| v == "1");
    let mut lent: Option<(Vec<memra_engine::cache::KvLend>, Vec<Vec<u8>>)> = None;
    let mut lent_turns = 0usize;
    // The previous turn's oracle input and how many tokens the session emitted from it, for the
    // continuation burst's reference.
    let mut prev_turn: Option<(Vec<u32>, usize)> = None;
    // turn 4 = EMPTY suffix (pure continuation burst, the serve pattern): must equal the
    // continuation of the previous turn's oracle (`prev_turn`), not a cold prime of the rows
    // the session's verify wrote.
    let turn4: Vec<u32> = Vec::new();
    // The harness tracks the TRUE context itself (suffixes + every emitted token) instead of
    // deriving it from session internals: since pending-carry (2026-08-01), `sess.committed`
    // legitimately lags `out` by one carried token at a burst boundary, so the old
    // `committed.len() - out.len()` prefix derivation no longer holds — and an oracle that
    // doesn't read the system under test's internals is stronger anyway.
    let mut hist: Vec<u32> = Vec::new();
    for (i, suffix) in [turn1, turn2, turn3, turn4].iter().enumerate() {
        hist.extend_from_slice(suffix);
        // What the session primes this turn, and from where.
        let (prime, resume): (Vec<u32>, String) = if suffix.is_empty() {
            (Vec::new(), "continuation burst".into())
        } else if !recurrent {
            (
                suffix.clone(),
                if i == 0 { "cold" } else { "parked end" }.into(),
            )
        } else if i == 0 {
            (hist.clone(), "cold".into())
        } else {
            match rewind_refusal(&sess, hist.len(), grid) {
                None => {
                    // memra#1221 review: rows past the checkpoint lent to a reader, held through
                    // the turn; their bytes are read now and after the turn.
                    lent = if lend_above {
                        let rows = sess.committed.len();
                        let lends = sess
                            .cache_ref()
                            .kv
                            .iter()
                            .flatten()
                            .filter(|l| l.len >= rows)
                            .flat_map(|l| {
                                [
                                    l.k.lend(rows * l.k_tok_bytes),
                                    l.v.lend(rows * l.v_tok_bytes),
                                ]
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let before = lends
                            .iter()
                            .map(|l| e.dtoh_u8(l))
                            .collect::<Result<Vec<_>, _>>()?;
                        println!(
                            "  lent {} planes, rows [0, {rows}), past the checkpoint",
                            lends.len()
                        );
                        Some((lends, before))
                    } else {
                        None
                    };
                    let pos = model
                        .spec_rewind_to_checkpoint(&e, &mut sess)?
                        .expect("the checkpoint passed rewind_refusal");
                    assert_eq!(
                        sess.committed[..],
                        hist[..pos],
                        "the session's rows through its checkpoint are the history's"
                    );
                    rewound_turns += 1;
                    (hist[pos..].to_vec(), format!("rewound to {pos}"))
                }
                Some(why) => {
                    sess = model.new_session(&e, 4096)?;
                    (hist.clone(), format!("cold ({why})"))
                }
            }
        };
        let split = if recurrent && !prime.is_empty() {
            let start = sess.committed.len();
            let boundary = stable_boundary(start, start + prime.len(), grid);
            sess.ckpt_at = boundary.or(sess.ckpt_at);
            boundary.map(|b| b - start)
        } else {
            None
        };
        let (out, _d, _a) = model.generate_spec_session_sampled_prime_split(
            &e,
            &mut sess,
            &prime,
            n_new,
            k,
            Some(SpecSampling::from_cli_env()),
            split,
            None,
        )?;
        // reference: the oracle's plain greedy over the FULL history (prior turns' suffixes +
        // outputs + this turn's suffix), re-primed from scratch: the independent oracle. A
        // continuation burst has no new prompt rows, so its reference is the previous turn's
        // oracle carried on past what that turn emitted.
        let reference = match (suffix.is_empty(), &prev_turn) {
            (true, Some((prev_hist, prev_len))) => {
                let long = model.generate_spec_oracle(&e, prev_hist, prev_len + out.len())?;
                long[(*prev_len).min(long.len())..].to_vec()
            }
            _ => model.generate_spec_oracle(&e, &hist, out.len())?,
        };
        if !suffix.is_empty() {
            prev_turn = Some((hist.clone(), out.len()));
        } else if let Some((_, prev_len)) = prev_turn.as_mut() {
            *prev_len += out.len();
        }
        hist.extend_from_slice(&out);
        let m = if !out.is_empty() && out == reference[..out.len().min(reference.len())] {
            "MATCH"
        } else {
            "MISMATCH"
        };
        if m == "MISMATCH" {
            ok = false;
        }
        println!(
            "turn {}: {resume}, {} tok generated, committed={} -> {m}",
            i + 1,
            out.len(),
            sess.committed.len()
        );
        if let Some((lends, before)) = lent.take() {
            let after = lends
                .iter()
                .map(|l| e.dtoh_u8(l))
                .collect::<Result<Vec<_>, _>>()?;
            let kept = after == before;
            lent_turns += 1;
            if !kept {
                ok = false;
            }
            println!(
                "  the lent rows after the turn: {}",
                if kept {
                    "unchanged"
                } else {
                    "CHANGED (a write landed below a lend)"
                }
            );
        }
        if m == "MISMATCH" {
            println!("  session:   {:?}", &out[..out.len().min(12)]);
            println!("  reference: {:?}", &reference[..reference.len().min(12)]);
        }
    }
    if recurrent && rewound_turns == 0 {
        println!("no turn rewound to a checkpoint: every turn primed cold, no boundary was tested");
        ok = false;
    }
    if lend_above && lent_turns == 0 {
        println!("MEMRA_GATE_LEND_ABOVE=1 but no rewind ran under a lend: nothing was tested");
        ok = false;
    }
    // A continuation burst whose first commit hook says stop (the admission yield at a burst's
    // entry) runs no round. The carried token stays pending and `committed` stays the cache
    // rows; the burst after it emits what the oracle emits.
    if let Some((prev_hist, prev_len)) = prev_turn.as_ref() {
        let carried = sess.has_pending();
        let mut stop = |_: &[u32]| false;
        let (idle, _d, _a) = model.generate_spec_session_sampled_prime_split(
            &e,
            &mut sess,
            &[],
            n_new,
            k,
            Some(SpecSampling::from_cli_env()),
            None,
            Some(&mut stop),
        )?;
        let (committed_idle, rows_idle) = (sess.committed.len(), sess.kv_position());
        let still_pending = sess.has_pending();
        // The session's list is the true history less the token still pending.
        let list_idle = sess.committed[..] == hist[..hist.len() - usize::from(still_pending)];
        let (next, _d, _a) = model.generate_spec_session_sampled_prime_split(
            &e,
            &mut sess,
            &[],
            n_new,
            k,
            Some(SpecSampling::from_cli_env()),
            None,
            None,
        )?;
        let mut got = idle.clone();
        got.extend_from_slice(&next);
        let long = model.generate_spec_oracle(&e, prev_hist, prev_len + got.len())?;
        let reference = &long[(*prev_len).min(long.len())..];
        let text_ok = !next.is_empty() && got[..] == reference[..got.len().min(reference.len())];
        hist.extend_from_slice(&got);
        let list_next = sess.committed[..] == hist[..hist.len() - usize::from(sess.has_pending())];
        let rows_ok = committed_idle == rows_idle
            && sess.committed.len() == sess.kv_position()
            && list_idle
            && list_next;
        let m = if carried && still_pending && idle.is_empty() && rows_ok && text_ok {
            "MATCH"
        } else {
            ok = false;
            "MISMATCH"
        };
        println!(
            "no-round burst: carried token before {carried}, after {still_pending}, {} tok emitted \
             by it, committed={committed_idle} cache rows={rows_idle}; next burst {} tok, \
             committed={} cache rows={}; committed is the history {list_idle}/{list_next} -> {m}",
            idle.len(),
            next.len(),
            sess.committed.len(),
            sess.kv_position()
        );
        if !carried {
            println!("the session held no carried token before the no-round burst: nothing tested");
        }
        if !text_ok {
            println!("  session:   {:?}", &got[..got.len().min(12)]);
            println!("  reference: {:?}", &reference[..reference.len().min(12)]);
        }
    }
    // A continuation burst whose commit hook says stop after its second slice (memra#1207: the
    // server ends a burst at the round that completes a stop string, and parks the session for
    // the next turn). The burst ends at that round with `committed` equal to the cache rows and
    // to the history less the pending token, and the burst after it emits what the oracle emits.
    if let Some((prev_hist, _)) = prev_turn.as_ref() {
        // Everything the session emitted since the turn the oracle continues.
        let since = hist.len() - prev_hist.len();
        let mut slices = 0usize;
        let mut stop_after_two = |s: &[u32]| {
            if !s.is_empty() {
                slices += 1;
            }
            slices < 2
        };
        let (cut, _d, _a) = model.generate_spec_session_sampled_prime_split(
            &e,
            &mut sess,
            &[],
            n_new,
            k,
            Some(SpecSampling::from_cli_env()),
            None,
            Some(&mut stop_after_two),
        )?;
        let (committed_cut, rows_cut) = (sess.committed.len(), sess.kv_position());
        let pending_cut = sess.has_pending();
        let mut hist_cut = hist.clone();
        hist_cut.extend_from_slice(&cut);
        let list_cut = sess.committed[..] == hist_cut[..hist_cut.len() - usize::from(pending_cut)];
        let (next, _d, _a) = model.generate_spec_session_sampled_prime_split(
            &e,
            &mut sess,
            &[],
            n_new,
            k,
            Some(SpecSampling::from_cli_env()),
            None,
            None,
        )?;
        let mut got = cut.clone();
        got.extend_from_slice(&next);
        let long = model.generate_spec_oracle(&e, prev_hist, since + got.len())?;
        let reference = &long[since.min(long.len())..];
        let text_ok = !next.is_empty() && got[..] == reference[..got.len().min(reference.len())];
        hist.extend_from_slice(&got);
        let list_next = sess.committed[..] == hist[..hist.len() - usize::from(sess.has_pending())];
        let rows_ok = committed_cut == rows_cut
            && sess.committed.len() == sess.kv_position()
            && list_cut
            && list_next;
        // The hook must have stopped the burst: two slices published and fewer tokens than the
        // burst's target, or the check tested nothing.
        let stopped = slices == 2 && cut.len() < n_new;
        let m = if stopped && rows_ok && text_ok {
            "MATCH"
        } else {
            ok = false;
            "MISMATCH"
        };
        println!(
            "stopped burst: the hook said stop after {slices} slices, {} of {n_new} tok emitted, \
             pending {pending_cut}, committed={committed_cut} cache rows={rows_cut}; next burst \
             {} tok, committed={} cache rows={}; committed is the history {list_cut}/{list_next} \
             -> {m}",
            cut.len(),
            next.len(),
            sess.committed.len(),
            sess.kv_position()
        );
        if !stopped {
            println!("the hook's stop did not end the burst early: nothing tested");
        }
        if !text_ok {
            println!("  session:   {:?}", &got[..got.len().min(12)]);
            println!("  reference: {:?}", &reference[..reference.len().min(12)]);
        }
    }
    // The sampled carry: one seeded, frequency-penalised request on three sessions: in one
    // burst (the reference: no boundary inside it), burst after burst, and burst after burst
    // with a no-round burst after each.
    {
        let sp = SpecSampling {
            temp: 1.0,
            seed: 4242,
            penalty_last_n: 8192,
            penalty_freq: 0.4,
            ..SpecSampling::GREEDY
        };
        let run = |burst: usize,
                   idle_bursts: bool|
         -> Result<(Vec<u32>, bool, bool), Box<dyn std::error::Error>> {
            let mut s = model.new_session(&e, 4096)?;
            let (mut all, mut carried, mut rows_ok) = (Vec::new(), true, true);
            let mut prime: &[u32] = &identity_prompt;
            while all.len() < 3 * n_new {
                let (out, _d, _a) = model.generate_spec_session_sampled_prime_split(
                    &e,
                    &mut s,
                    prime,
                    burst,
                    k,
                    Some(sp),
                    None,
                    None,
                )?;
                if out.is_empty() {
                    return Err("a sampled burst emitted nothing".into());
                }
                prime = &[];
                all.extend_from_slice(&out);
                carried &= s.has_pending();
                rows_ok &= s.committed.len() == s.kv_position();
                if idle_bursts {
                    let mut stop = |_: &[u32]| false;
                    let (idle, _d, _a) = model.generate_spec_session_sampled_prime_split(
                        &e,
                        &mut s,
                        &[],
                        n_new,
                        k,
                        Some(sp),
                        None,
                        Some(&mut stop),
                    )?;
                    all.extend_from_slice(&idle);
                    carried &= s.has_pending();
                    rows_ok &= s.committed.len() == s.kv_position();
                }
            }
            Ok((all, carried, rows_ok))
        };
        let (single, _, rows_s) = run(3 * n_new, false)?;
        let (straight, carried_a, rows_a) = run(n_new, false)?;
        let (idled, carried_b, rows_b) = run(n_new, true)?;
        let n = single.len().min(straight.len()).min(idled.len());
        let same = single[..n] == straight[..n] && single[..n] == idled[..n];
        let m = if same && carried_a && carried_b && rows_s && rows_a && rows_b {
            "MATCH"
        } else {
            ok = false;
            "MISMATCH"
        };
        println!(
            "sampled carry: {} tok in one burst, {} burst after burst, {} with no-round bursts, \
             first {n} equal {same}; every burst ended on a carried token {carried_a}/{carried_b}; \
             committed == cache rows {rows_s}/{rows_a}/{rows_b} -> {m}",
            single.len(),
            straight.len(),
            idled.len()
        );
        if !same {
            let at = (0..n)
                .find(|&i| single[i] != straight[i] || single[i] != idled[i])
                .unwrap_or(n);
            println!("  first difference at generated position {at}");
        }
    }
    println!(
        "{}",
        if ok {
            "SESSION GATE: ALL TURNS MATCH"
        } else {
            "SESSION GATE: FAIL"
        }
    );
    std::process::exit(if ok { 0 } else { 1 });
}
