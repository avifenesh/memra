//! gemma-spec-session-gate: the burst-boundary exactness battery for the served gemma4
//! spec session (lane/gemma-batched stage 1, 2026-08-16).
//!
//! The Q38 serve-spec lane's bug class lived at burst boundaries (pending-carry,
//! empty-suffix continuation, demote handoff). These cases are BANKED HERE as explicit
//! gates, written before the session refactor per the coordinator's directive:
//!
//!   gate1 — session == one-shot, per burst width w in {1, K-1, K, K+1, 32}: the emitted
//!           stream, reassembled across bursts, must be BYTE-IDENTICAL (u32 token ids) to
//!           `generate_spec_gemma`'s stream. w=1 makes EVERY round a burst boundary; the
//!           K-neighborhood widths straddle the round size (the pending-carry class).
//!   gate2 — session == plain greedy (the spec exactness law): full-stream agreement.
//!   gate3 — DEMOTE MID-BURST: burst(16), `into_demoted()`, continue PLAIN eager decode
//!           from the handed-off (cache, pending) — the combined stream must equal plain
//!           greedy. Proves the handoff state is genuinely resumable (stage-2 de-risk),
//!           gated, not assumed-clean.
//!   invariants — after every arm: cache.pos == committed.len() (rows == tokens), and the
//!           session's committed stream extends the emitted stream (overshoot only).
//!
//! Trim-adapt isolation: `generate_spec_gemma`'s tail APPENDS to the `<ranks>.learned`
//! sidecar, and the trim state changes drafting behavior — so every arm loads its
//! GemmaDraft from a FRESH temp copy of the ranks file (identical initial trim state,
//! sidecar writes discarded). Without this the arms legitimately diverge and the gate
//! measures rank-file mutation, not burst boundaries.
//!
//! Usage: gemma-spec-session-gate <model.gguf> <prompt token ids...>
//!   env: MEMRA_DRAFT=<assistant.gguf> (required)  MEMRA_SPEC=K (default 5)
//!        MEMRA_NGEN=N (default 128)  MEMRA_GEMMA_DRAFT_RANKS=<ranks> (optional)

use memra_engine::gemma_spec::GemmaDraft;
use memra_engine::hybrid::HybridModel;

fn load_draft_isolated(
    e: &memra_engine::Engine,
    dpath: &str,
    ranks_src: Option<&str>,
    arm: &str,
) -> Result<GemmaDraft, Box<dyn std::error::Error>> {
    if let Some(src) = ranks_src {
        let dir = std::env::temp_dir().join(format!("g4-sess-gate-{}-{arm}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let dst = dir.join("ranks.txt");
        std::fs::copy(src, &dst)?;
        // pre-existing sidecar travels too (identical initial trim state across arms).
        let side = format!("{src}.learned");
        if std::path::Path::new(&side).exists() {
            std::fs::copy(&side, dir.join("ranks.txt.learned"))?;
        }
        // single-threaded bin; load reads the env at call time.
        unsafe { std::env::set_var("MEMRA_GEMMA_DRAFT_RANKS", dst.to_str().unwrap()) };
    }
    let dg = memra_gguf::GgufFile::open(dpath)?;
    Ok(GemmaDraft::load(e, &dg)?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: gemma-spec-session-gate <model.gguf> <ids...>");
    let toks: Vec<u32> = args.filter_map(|s| s.parse().ok()).collect();
    assert!(!toks.is_empty(), "need prompt token ids");
    let dpath = std::env::var("MEMRA_DRAFT").expect("MEMRA_DRAFT=<assistant.gguf> required");
    let k: usize = std::env::var("MEMRA_SPEC")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let n_new: usize = std::env::var("MEMRA_NGEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);
    let ranks_src = std::env::var("MEMRA_GEMMA_DRAFT_RANKS").ok();
    let ranks = ranks_src.as_deref();

    let e = memra_engine::Engine::new(0)?;
    let g = memra_gguf::GgufFile::open(&path)?;
    let model = HybridModel::load(&e, &g)?;
    let ctx = toks.len() + n_new + 4 * (k + 2); // overshoot headroom
    let mut fails = 0usize;
    let mut check = |name: &str, ok: bool| {
        if ok {
            println!("  ok: {name}");
        } else {
            println!("  FAIL: {name}");
            fails += 1;
        }
    };

    // ---- references ----
    let plain = model.generate(&e, &toks, n_new)?;
    let mut d_ref = load_draft_isolated(&e, &dpath, ranks, "oneshot")?;
    let oneshot = model.generate_spec_gemma(&e, &mut d_ref, &toks, n_new, k, &[])?;
    let agree = plain
        .iter()
        .zip(&oneshot)
        .take_while(|(a, b)| a == b)
        .count();
    println!(
        "references: plain {} toks, one-shot {} toks, agreement {}/{}",
        plain.len(),
        oneshot.len(),
        agree,
        plain.len().min(oneshot.len())
    );
    check(
        "one-shot == plain (precondition, the existing exactness law)",
        agree == plain.len().min(oneshot.len()),
    );

    // ---- gate1/gate2: session at every banked burst width ----
    let widths = [1usize, k.saturating_sub(1).max(1), k, k + 1, 32];
    for &w in &widths {
        let mut d = load_draft_isolated(&e, &dpath, ranks, &format!("w{w}"))?;
        let mut sess = model.gemma_spec_session_new(&e, &mut d, &toks, ctx)?;
        let mut stream: Vec<u32> = Vec::new();
        let mut bursts = 0usize;
        while stream.len() < n_new {
            let (chunk, _dr, _ac) =
                model.gemma_spec_session_burst(&e, &mut d, &mut sess, w, k, &[])?;
            if chunk.is_empty() {
                break;
            }
            stream.extend_from_slice(&chunk);
            bursts += 1;
        }
        let n = n_new.min(stream.len()).min(oneshot.len());
        let same_os = stream[..n] == oneshot[..n];
        let np = n_new.min(stream.len()).min(plain.len());
        let same_pl = stream[..np] == plain[..np];
        println!(
            "w={w}: {bursts} bursts, {} toks (committed {}, cache.pos {})",
            stream.len(),
            sess.committed.len() - sess.prompt_len,
            sess.cache.pos
        );
        check(
            &format!("gate1 w={w}: session == one-shot ({n} toks)"),
            same_os,
        );
        check(
            &format!("gate2 w={w}: session == plain ({np} toks)"),
            same_pl,
        );
        check(
            &format!("invariant w={w}: cache rows == committed"),
            sess.cache.pos == sess.committed.len(),
        );
        // committed extends the emitted stream (overshoot + post-EOS rows only).
        let emitted = &sess.committed[sess.prompt_len..];
        check(
            &format!("invariant w={w}: emitted stream is a committed prefix"),
            emitted.len() >= stream.len() && emitted[..stream.len()] == stream[..],
        );
        if !same_os {
            let div = stream
                .iter()
                .zip(&oneshot)
                .take_while(|(a, b)| a == b)
                .count();
            println!(
                "    diverges at {div}: session {:?} one-shot {:?}",
                &stream[div..(div + 6).min(stream.len())],
                &oneshot[div..(div + 6).min(oneshot.len())]
            );
        }
    }

    // ---- gate3: demote mid-burst -> plain continuation == plain greedy ----
    {
        let mut d = load_draft_isolated(&e, &dpath, ranks, "demote")?;
        let mut sess = model.gemma_spec_session_new(&e, &mut d, &toks, ctx)?;
        let (head, _dr, _ac) = model.gemma_spec_session_burst(&e, &mut d, &mut sess, 16, k, &[])?;
        check(
            "gate3 precondition: cache rows == committed at demote",
            sess.cache.pos == sess.committed.len(),
        );
        let (mut cache, pending, committed) = sess.into_demoted();
        // plain continuation: emit pending, feed it, argmax-loop (the eager plain program).
        let mut stream: Vec<u32> = committed[toks.len()..].to_vec();
        let mut last = pending;
        while stream.len() < n_new {
            stream.push(last);
            if stream.len() >= n_new {
                break;
            }
            let (l, _h) = model.decode_step_h(&e, last, &mut cache)?;
            last = memra_engine::forward::argmax(&l) as u32;
        }
        let n = n_new.min(stream.len()).min(plain.len());
        let same = stream[..n] == plain[..n];
        println!(
            "demote: {} spec toks (visible {}), then plain continuation to {n}",
            committed.len() - toks.len(),
            head.len()
        );
        check(
            &format!("gate3: spec-16 -> demote -> plain == plain greedy ({n} toks)"),
            same,
        );
        if !same {
            let div = stream
                .iter()
                .zip(&plain)
                .take_while(|(a, b)| a == b)
                .count();
            println!(
                "    diverges at {div}: demoted {:?} plain {:?}",
                &stream[div..(div + 6).min(stream.len())],
                &plain[div..(div + 6).min(plain.len())]
            );
        }
    }

    if fails == 0 {
        println!("ALL GREEN: gemma spec session burst-boundary battery");
        Ok(())
    } else {
        println!("{fails} FAILURES");
        std::process::exit(1);
    }
}
