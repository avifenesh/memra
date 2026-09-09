//! CONTINUATION gate: priming a prompt in one call must equal priming it in pieces.
//!
//! A restored turn primes only the suffix onto state it did not compute in this call. The server
//! evidence said the calibrated artifact's turn-4 prime row differs from the cold one while the
//! served artifact's is identical on all four turns, which points at the prime path rather than
//! at the cache. This removes the server, the prefix cache, the drafter and the boundary capture
//! entirely: the same tokens, one cache, either one call or several.
//!
//! If arm B diverges from arm A here, the fault is the engine's own prime continuation under the
//! activation program, and the restore machinery was only the thing that exercised it.
//!
//! usage: qwen-a4-continuation-gate <model.gguf> <prompt.txt> [total] [splits...]
use memra_engine::Engine;
use memra_engine::hybrid::HybridModel;
use memra_gguf::GgufFile;

fn digest(row: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in row {
        for b in v.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: qwen-a4-continuation-gate <model.gguf> <prompt.txt> [total] [splits...]");
    let prompt = args.next().expect("a prompt file");
    let total: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(9296);
    let splits: Vec<usize> = {
        let rest: Vec<usize> = args.filter_map(|v| v.parse().ok()).collect();
        if rest.is_empty() {
            vec![48, 49, 151, 80, 16, 17]
        } else {
            rest
        }
    };

    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let tok = memra_tokenizer::Tokenizer::from_gguf(&g)?;
    let model = HybridModel::load(&e, &g)?;
    println!(
        "{path}: activation program {}",
        if model.cfg.prefill_activation.is_some() {
            "PRESENT"
        } else {
            "absent"
        }
    );
    let mut ids = tok.encode(&std::fs::read_to_string(&prompt)?, true);
    assert!(
        ids.len() >= total,
        "prompt has {} tokens, need {total}",
        ids.len()
    );
    ids.truncate(total);

    // ARM A: one call.
    let mut cache = memra_engine::pp::new_cache(&e, &model.cfg, total + 64)?;
    let (whole, _, _) = model.prime_cache(&e, &ids, &mut cache, 0)?;
    let reference = digest(&whole);
    println!("one call over {total} tokens: logits_sha={reference:016x}");

    let mut failures = 0usize;
    for &tail in &splits {
        if tail >= total {
            continue;
        }
        let head = total - tail;
        // ARM B: the same tokens, one cache, two calls. The second call is the "suffix prime".
        let mut split_cache = memra_engine::pp::new_cache(&e, &model.cfg, total + 64)?;
        model.prime_cache(&e, &ids[..head], &mut split_cache, 0)?;
        let (rest, _, _) = model.prime_cache(&e, &ids[head..], &mut split_cache, 0)?;
        let got = digest(&rest);
        let verdict = if got == reference { "ok" } else { "DIFFERS" };
        println!("  {head} + {tail}: logits_sha={got:016x} {verdict}");
        if got != reference {
            failures += 1;
        }
    }
    if failures != 0 {
        eprintln!(
            "A4 CONTINUATION GATE: {failures} of {} splits differ from the one-call prime",
            splits.len()
        );
        std::process::exit(1);
    }
    println!("A4 CONTINUATION GATE: PASS");
    Ok(())
}
