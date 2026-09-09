//! RUNTIME dispatch gate for the calibrated Qwen prefill program (qwen35-prefill-nvfp4-a4-v1).
//!
//! The unit tests in the qwen35 pack prove the POLICY (`select_activation` is prefill-only, a
//! scale-less artifact never selects A4). They cannot prove the WIRING: the prefill phase is
//! hand-threaded through the prime call sites, and a missed one fails safe but silent — the
//! artifact would declare a 400-linear activation program while the engine ran it on fewer.
//!
//! This gate closes that gap with the A4 launch counter:
//!
//!   1. a prime issues a NON-ZERO, exact multiple of 400 A4 GEMMs (one per stamped projection
//!      per prime chunk), and the per-chunk count is exactly 400;
//!   2. decode issues ZERO at any batch size;
//!   3. speculative generation issues exactly what its own prime issued, so target verification
//!      added none;
//!   4. the SERVED artifact (no activation program) issues ZERO through the same prime.
//!
//! Arm 1 is what a dropped call site breaks: revert any one of them and the count falls below
//! 400 per chunk. Arm 4 is the legacy control. Arm 2/3 are the phase isolation the owner asked
//! for, measured on the engine rather than asserted on the enum.
//!
//! Usage: qwen-a4-dispatch-gate <calibrated.gguf> <served.gguf> [prime_tokens]
use memra_engine::Engine;
use memra_engine::hybrid::HybridModel;
use memra_engine::mmq_ffi::{a4_prefill_launches_reset, a4_prefill_slots_reset};
use memra_gguf::GgufFile;

const PROGRAM_LINEARS: u64 = 400;

fn load(e: &Engine, path: &str) -> Result<HybridModel, Box<dyn std::error::Error>> {
    let g = GgufFile::open(path)?;
    HybridModel::load(e, &g)
}

/// Deterministic, tokenizer-free prompt: this gate counts GEMM launches, so the token VALUES are
/// irrelevant as long as they are in range and the length is stable.
fn prompt(n: usize, vocab: u32) -> Vec<u32> {
    (0..n)
        .map(|i| ((i * 7919 + 13) as u32) % vocab.min(30000))
        .collect()
}

fn fail(what: &str) -> ! {
    eprintln!("FAIL: {what}");
    std::process::exit(1)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let calibrated = args
        .next()
        .expect("usage: qwen-a4-dispatch-gate <calibrated.gguf> <served.gguf> [prime_tokens]");
    let served = args
        .next()
        .expect("the served artifact is the legacy control arm");
    let t: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(4096);

    let e = Engine::new(0)?;
    let model = load(&e, &calibrated)?;
    let program = model
        .cfg
        .prefill_activation
        .clone()
        .unwrap_or_else(|| fail("the calibrated artifact declares no activation program"));
    println!("program: {program:?}");
    if program.scales().len() as u64 != PROGRAM_LINEARS {
        fail("the activation program is not 400 linears");
    }

    let ids = prompt(t, model.cfg.n_vocab);

    // ---- arm 1: the prime runs the whole declared program, once per chunk ----
    let names = program.slot_names();
    let mut cache = memra_engine::pp::new_cache(&e, &model.cfg, ids.len() + 64)?;
    a4_prefill_launches_reset();
    a4_prefill_slots_reset();
    model.prime_cache(&e, &ids, &mut cache, 0)?;
    let primed = a4_prefill_launches_reset();
    let slots = a4_prefill_slots_reset();
    if primed == 0 {
        fail("the prime issued ZERO calibrated A4 GEMMs: dispatch is not wired");
    }
    // Every stamped projection must run the SAME number of times: that count is the number of
    // prime chunks, and any projection below it still has a call site on the W4A8 walk. A bare
    // total cannot say which one, and the total is what a missed call site quietly changes.
    let chunks = *slots.iter().max().expect("the program has 400 slots");
    let missing: Vec<(&str, u64)> = names
        .iter()
        .zip(slots.iter())
        .filter(|(_, ran)| **ran != chunks)
        .map(|(name, ran)| (*name, *ran))
        .collect();
    if !missing.is_empty() {
        eprintln!(
            "{} of {PROGRAM_LINEARS} projections did not run the program {chunks} times:",
            missing.len()
        );
        for (name, ran) in missing.iter().take(24) {
            eprintln!("  {name}: {ran} of {chunks}");
        }
        if missing.len() > 24 {
            eprintln!("  ... and {} more", missing.len() - 24);
        }
        fail("at least one stamped projection is missing a prefill call site");
    }
    if primed != PROGRAM_LINEARS * chunks {
        fail(&format!(
            "the prime issued {primed} A4 GEMMs but only {} were attributed to slots",
            PROGRAM_LINEARS * chunks
        ));
    }
    println!(
        "PASS arm1 prime: {primed} A4 GEMMs over {t} tokens = all {PROGRAM_LINEARS} projections x {chunks} prime chunks"
    );

    // ---- arm 2: decode is W4A8 at every batch size ----
    for batch in [1usize, 4] {
        let mut caches: Vec<memra_engine::cache::Cache> = (0..batch)
            .map(|_| memra_engine::pp::new_cache(&e, &model.cfg, ids.len() + 64))
            .collect::<Result<_, _>>()?;
        for c in caches.iter_mut() {
            model.prime_cache(&e, &ids, c, 0)?;
        }
        a4_prefill_launches_reset();
        let mut refs: Vec<&mut memra_engine::cache::Cache> = caches.iter_mut().collect();
        for step in 0..8 {
            let tokens = vec![ids[step % ids.len()]; batch];
            model.decode_step_batch(&e, &tokens, &mut refs)?;
        }
        let decoded = a4_prefill_launches_reset();
        if decoded != 0 {
            fail(&format!(
                "decode at batch {batch} issued {decoded} calibrated A4 GEMMs: decode must stay W4A8"
            ));
        }
        println!("PASS arm2 decode b={batch}: 0 A4 GEMMs over 8 steps");
    }

    // ---- arm 3: speculative verification adds none over its own prime ----
    if model.mtp.is_some() {
        let short = prompt(t.min(1024), model.cfg.n_vocab);
        let mut warm = memra_engine::pp::new_cache(&e, &model.cfg, short.len() + 64)?;
        a4_prefill_launches_reset();
        model.prime_cache(&e, &short, &mut warm, 0)?;
        let prime_only = a4_prefill_launches_reset();
        drop(warm);
        model.generate_spec(&e, &short, 32, 4)?;
        let with_spec = a4_prefill_launches_reset();
        if with_spec != prime_only {
            fail(&format!(
                "speculative generation issued {with_spec} A4 GEMMs but its prime alone issues \
                 {prime_only}: target verification took the prefill program"
            ));
        }
        println!(
            "PASS arm3 spec: {with_spec} A4 GEMMs, identical to the bare prime ({prime_only})"
        );
    } else {
        fail("the calibrated artifact lost its MTP head");
    }

    // ---- arm 4: the served artifact is untouched ----
    drop(model);
    let legacy = load(&e, &served)?;
    if legacy.cfg.prefill_activation.is_some() {
        fail("the SERVED artifact declares an activation program: wrong file");
    }
    let legacy_ids = prompt(t, legacy.cfg.n_vocab);
    let mut legacy_cache = memra_engine::pp::new_cache(&e, &legacy.cfg, legacy_ids.len() + 64)?;
    a4_prefill_launches_reset();
    legacy.prime_cache(&e, &legacy_ids, &mut legacy_cache, 0)?;
    let legacy_primed = a4_prefill_launches_reset();
    if legacy_primed != 0 {
        fail(&format!(
            "the served scale-less artifact issued {legacy_primed} calibrated A4 GEMMs"
        ));
    }
    println!("PASS arm4 legacy: served artifact primed {t} tokens with 0 A4 GEMMs");

    println!("A4 DISPATCH GATE: PASS");
    Ok(())
}
