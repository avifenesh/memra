//! SESSION EXACTNESS GATE: multi-turn generate_spec_session must produce, at every turn, the
//! IDENTICAL tokens plain greedy generate() produces when primed with the session's full
//! committed history. Verify guarantees per-turn exactness; this gate pins the turn BOUNDARY
//! (the resume a new turn takes, its suffix prime and the cross-turn draft-KV/pairing state).
//!
//! The resume a turn takes is the one the served engine takes (memra#1146). An attention-only
//! trunk resumes at its parked end. A recurrent (GDN) trunk does not: the rows past the prompt
//! were written by verify and decode, off the prime grid, and a suffix primed from there starts
//! from state a cold prime never computes (the grid law, memra#1091 and memra#1093, the worker's
//! `spec_pool_grid_refusal`). The gate used to run that parked-end resume on every trunk; on
//! Qwen3.8-27B it matched at K=2 with 24 tokens per turn and left the reference at K=3 with 48
//! (GB10 and RTX 5090), by the margin of whichever argmax the off-grid state moved. On a
//! recurrent trunk the gate now refuses the parked-end resume and takes the served one: rewind
//! to the session's turn checkpoint when it sits on the grid and the prompt extends it by the
//! prime floor, else a cold prime in a fresh session. Each prime stops at the last grid boundary
//! that leaves the prime floor behind it and captures the turn checkpoint there, as serving's
//! stable boundary does.
//!
//! usage: session-gate <model.gguf>   (env: MEMRA_SPEC_K, MEMRA_NGEN per turn)
use memra_engine::Engine;
use memra_engine::forward::argmax;
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

/// Greedy oracle for the target numeric class. Step35 and the qwen35 family serve B=1 on the
/// batched program (concurrency cannot steer bytes), so comparing their speculative verifiers
/// to the older eager `generate` class is a false red at near ties: MoE learned it 2026-08-14
/// AM, the dense hybrid the same day on Qwen3.8 (an eager oracle here would fail the aligned
/// serving-class verify on exactly the near-ties the fix removed). Reuse the production batch
/// step, including stage-owned PP caches. A sharded PP target must likewise stay on a split
/// target step: `generate` may select the unsplit device-counter loop, which correctly refuses
/// remote weights rather than peer-reading them. Other single-device families keep the
/// longstanding generate oracle.
fn plain_oracle(
    model: &HybridModel,
    e: &Engine,
    prompt: &[u32],
    max_new: usize,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let batched_class = model.uses_sliding_gated_moe_program()
        || memra_engine::plan_backend::gdn_dspark_compatible(&model.plan);
    let pp_class = memra_engine::pp::pp_cuts(model.layers.len()).is_some();
    if !batched_class && !pp_class {
        return model.generate(e, prompt, max_new);
    }
    let mut cache = memra_engine::pp::new_cache(e, &model.cfg, prompt.len() + max_new + 8)?;
    let t_prime = std::time::Instant::now();
    let mut logits = if prompt.len() >= memra_engine::hybrid_forward::PRIME_MIN_T
        && std::env::var("MEMRA_PRIME_TOKENWISE").is_err()
        && !e.frozen_cpu_experts_prefer_tokenwise_prime()
    {
        model.prime_cache(e, prompt, &mut cache, 0)?.0
    } else {
        let mut row = Vec::new();
        for &token in prompt {
            // The live GDN worker primes cold sub-floor prompts eagerly even though
            // generated tokens use the batched serving class (#918). Keep the oracle
            // tied to that request program, rather than copying the speculative prime.
            row = if batched_class
                && !(memra_engine::plan_backend::gdn_dspark_compatible(&model.plan)
                    && prompt.len() < memra_engine::hybrid_forward::PRIME_MIN_T)
            {
                let mut caches = [&mut cache];
                model.decode_step_batch(e, &[token], &mut caches)?.remove(0)
            } else {
                model.decode_step_h(e, token, &mut cache)?.0
            };
        }
        row
    };
    e.stream().synchronize()?;
    memra_engine::PRIME_NANOS.store(
        t_prime.elapsed().as_nanos() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    let mut out = Vec::with_capacity(max_new);
    for _ in 0..max_new {
        let token = argmax(&logits) as u32;
        out.push(token);
        logits = if batched_class {
            let mut caches = [&mut cache];
            model.decode_step_batch(e, &[token], &mut caches)?.remove(0)
        } else {
            model.decode_step_h(e, token, &mut cache)?.0
        };
    }
    Ok(out)
}


/// One plain step of the serving class against one verify row, from twin primed caches: the
/// logits must be the same bits.
fn step_identity(model: &HybridModel, e: &Engine, prompt: &[u32]) -> Result<bool, Box<dyn std::error::Error>> {
    let n_vocab = model.cfg.n_vocab as usize;
    let batched_class = model.uses_sliding_gated_moe_program()
        || memra_engine::plan_backend::gdn_dspark_compatible(&model.plan);
    let mut a = memra_engine::pp::new_cache(e, &model.cfg, prompt.len() + 16)?;
    let la = model.prime_cache(e, prompt, &mut a, 0)?.0;
    let mut b = memra_engine::pp::new_cache(e, &model.cfg, prompt.len() + 16)?;
    let _ = model.prime_cache(e, prompt, &mut b, 0)?;
    let tok = argmax(&la[..n_vocab]) as u32;
    let pos = a.pos;
    let plain = if batched_class {
        let mut caches = [&mut a];
        model.decode_step_batch(e, &[tok], &mut caches)?.remove(0)
    } else {
        model.decode_step_h(e, tok, &mut a)?.0
    };
    let verify = {
        let _exact = e.exact_scope(true);
        model.decode_step_t_h(e, &[tok], pos, &mut b)?.0
    };
    let differing = (0..n_vocab).filter(|&i| plain[i].to_bits() != verify[i].to_bits()).count();
    let worst = (0..n_vocab).map(|i| (plain[i] - verify[i]).abs()).fold(0f32, f32::max);
    println!(
        "step identity ({} tokens primed, {} plain step vs one verify row): {differing} of {n_vocab} logits differ, max |diff| {worst}",
        prompt.len(),
        if batched_class { "batched" } else { "eager" }
    );
    Ok(differing == 0)
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

    // Three turns of real-ish token suffixes (numeric ids are fine: the gate is exactness,
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
    {
        let p: Vec<u32> = (101..229).chain(350..415).collect();
        if !step_identity(&model, &e, &p)? { ok = false; }
    }
    // turn 4 = EMPTY suffix (pure continuation burst: the serve pattern): must equal greedy
    // over the full committed history, same oracle as the other turns.
    let turn4: Vec<u32> = Vec::new();
    // The harness tracks the TRUE context itself (suffixes + every emitted token) instead of
    // deriving it from session internals: since pending-carry (2026-08-01), `sess.committed`
    // legitimately lags `out` by one carried token at a burst boundary, so the old
    // `committed.len() - out.len()` prefix derivation no longer holds: and an oracle that
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
                    let pos = model
                        .spec_rewind_to_checkpoint(&e, &mut sess)?
                        .expect("the checkpoint passed rewind_refusal");
                    assert_eq!(
                        sess.committed[..],
                        hist[..pos],
                        "the session's rows through its checkpoint are the history's"
                    );
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
        // reference: plain greedy over the FULL history (prior turns' suffixes + outputs +
        // this turn's suffix). generate() re-primes from scratch: the independent oracle.
        let reference = plain_oracle(&model, &e, &hist, out.len())?;
        let eager = model.generate(&e, &hist, out.len())?;
        if eager != reference { println!("  (eager generate() differs from the serving-class oracle at {:?})", (0..eager.len().min(reference.len())).find(|&i| eager[i] != reference[i])); }
        hist.extend_from_slice(&out);
        let m = if out == reference[..out.len().min(reference.len())] {
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
        if m == "MISMATCH" {
            println!("  session:   {:?}", &out[..out.len().min(12)]);
            println!("  reference: {:?}", &reference[..reference.len().min(12)]);
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
