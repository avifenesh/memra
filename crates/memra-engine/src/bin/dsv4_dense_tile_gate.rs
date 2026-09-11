//! Dense wide-prefill TILING INVARIANT (memra #471): a wide transaction's value
//! does not depend on how the dense entry points cut it into tiles.
//!
//! This is a check, not a door. The tile width is `DSV4_TMAX` in the code and
//! nothing selects it at runtime, so there is no arm to switch and no seam to
//! switch it with. What is gated is the property the naked default rests on.
//!
//! Above `DSV4_TMAX` the four dense entry points decompose a transaction and
//! relaunch the registered-row kernel per tile. `M` is a template parameter that
//! decides only how many independent accumulators a block keeps in registers:
//! each `part[t]` advances over the same `i0` sequence, in the same order, and is
//! reduced by the same 128-leaf halving tree whatever `M` is. So a chunk of 32
//! rows, which takes the dispatch switch directly and never tiles, must be
//! byte-identical to a chunk of 64, which tiles as 2 x 32, and to 512, which
//! tiles as 16 x 32. That is what makes the tile width a free choice and the
//! `+3.94% / +4.10%` of memra #468 a same-class win rather than a new class.
//!
//! What makes it a check rather than a ritual:
//! - caller: `main`, every run, every width, nothing to opt into;
//! - non-vacuous input: a real >= 1025-token source prompt (padding refused), and
//!   the tiled-path counter must MOVE at every width above `DSV4_TMAX`, so a walk
//!   that never tiled cannot pass as one that did. Width 32 is asserted NOT to
//!   tile, which is what makes it the untiled reference rather than a fourth
//!   tiled arm;
//! - red arm: two, both every run. The comparator is handed a one-ULP change of a
//!   real digest and must refuse naming the byte, and the whole pipeline is
//!   re-run on a one-token-different prompt whose digest must differ.
use memra_engine::dsv4_gpu::{Dsv4Gpu, Dsv4SampleCfg, Dsv4Vt, dense_tile_counts_for_gate};
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::path::Path;
use std::time::Instant;

/// `DSV4_TMAX`. At or below it a dense call takes the dispatch switch directly.
const TMAX: usize = 32;
/// The width a customer request prefills at today; the one that used to tile by 8.
const SERVED_WIDTH: usize = 64;

fn floats(hash: &mut Sha256, values: &[f32], mask: bool) {
    hash.update((values.len() as u64).to_le_bytes());
    for value in values {
        assert!(value.is_finite() || (mask && *value == f32::NEG_INFINITY));
        hash.update(value.to_bits().to_le_bytes());
    }
}
fn classes(hash: &mut Sha256, values: Vec<(String, Vec<f32>)>) {
    assert!(!values.is_empty());
    for (name, values) in values {
        hash.update(name.as_bytes());
        floats(
            hash,
            &values,
            name.ends_with(".cmp_pend_score") || name.ends_with(".idx_pend_score"),
        );
    }
}

/// Refuse on the FIRST differing bit, and say where it is. A bare `assert_eq!` on
/// two hashes tells the reader something moved and nothing else.
fn same_bits(what: &str, reference: &[u8], candidate: &[u8]) -> Result<(), String> {
    if reference.len() != candidate.len() {
        return Err(format!(
            "{what}: digest length {} vs {}",
            reference.len(),
            candidate.len()
        ));
    }
    for (index, (a, b)) in reference.iter().zip(candidate).enumerate() {
        if a != b {
            let bit = (a ^ b).leading_zeros();
            return Err(format!(
                "{what}: first differing bit at byte {index}, bit {bit} ({a:#04x} vs {b:#04x})"
            ));
        }
    }
    Ok(())
}

struct Run {
    digest: Vec<u8>,
    seconds: f64,
    decompositions: u64,
    tiles: u64,
}

fn run(gpu: &Dsv4Gpu, prompt: &[u32], width: usize) -> Run {
    let before = dense_tile_counts_for_gate();
    let mut state = gpu
        .alloc_decode_state_for_transient(prompt.len() + 32, width)
        .expect("cache");
    let mut draft = gpu.dspark_alloc_state().expect("draft");
    let timer = Instant::now();
    let logits = gpu
        .dspark_prefill_prime_chunked(prompt, &mut state, &mut draft, width)
        .expect("prime");
    let seconds = timer.elapsed().as_secs_f64();

    let mut hash = Sha256::new();
    floats(&mut hash, &logits, false);
    classes(&mut hash, gpu.cache_classes(&state).expect("trunk classes"));
    classes(
        &mut hash,
        gpu.dspark_ring_classes(&draft).expect("draft classes"),
    );
    // The vendor-default sampled shape is what serves, so the invariant is asked
    // of a sampled continuation and not only of a prefill's logits.
    let cfg = Dsv4SampleCfg {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        seed: 20260910,
    };
    let mut verify = gpu.alloc_verify_state_for(state.capacity).expect("verify");
    let sampled = gpu
        .spec_sampled_batched_pen_restored(
            prompt,
            &logits,
            16,
            &mut state,
            &mut draft,
            &mut verify,
            usize::MAX,
            Dsv4Vt::Off,
            &cfg,
            None,
            None,
        )
        .expect("sampled spec");
    assert_eq!(sampled.tokens.len(), 16);
    assert!(!sampled.rounds.is_empty());
    for token in sampled.tokens {
        hash.update(token.to_le_bytes());
    }
    for round in sampled.rounds {
        for value in [
            round.start_pos,
            round.accepts,
            round.verified,
            round.t_batch,
            round.emitted,
        ] {
            hash.update((value as u64).to_le_bytes());
        }
        floats(&mut hash, &round.confidence, false);
    }
    classes(&mut hash, gpu.cache_classes(&state).expect("final trunk"));
    classes(
        &mut hash,
        gpu.dspark_ring_classes(&draft).expect("final draft"),
    );

    let after = dense_tile_counts_for_gate();
    let run = Run {
        digest: hash.finalize().to_vec(),
        seconds,
        decompositions: after[0] - before[0],
        tiles: after[1] - before[1],
    };
    println!(
        "RUN width={width} tokens={} seconds={:.6} decompositions={} tiles={}",
        prompt.len(),
        run.seconds,
        run.decompositions,
        run.tiles
    );
    run
}

fn main() {
    // Freeze the other dense-family doors so this gate exercises ONE property.
    // Process startup, before any model or worker thread exists.
    unsafe {
        std::env::set_var("MEMRA_DSV4_HC_DOT_SPLIT", "0");
        std::env::set_var("MEMRA_DSV4_DENSE_FAST", "0");
        std::env::set_var("MEMRA_DSV4_AR_PHASE", "0");
    }

    let args: Vec<String> = std::env::args().collect();
    assert_eq!(
        args.len(),
        3,
        "usage: dsv4_dense_tile_gate <model-dir> <real-source.txt>"
    );
    let dir = Path::new(&args[1]);
    let source = std::fs::read_to_string(&args[2]).expect("real source");
    let tokenizer = Tokenizer::from_hf_dir(dir).expect("tokenizer");
    let mut prompt = tokenizer.encode(&format!("Review this engine source:\n{source}"), true);
    assert!(prompt.len() >= 1025, "do not pad/repeat source");
    prompt.truncate(1025);
    println!(
        "SOURCE sha256={:x} tokens={}",
        Sha256::digest(source.as_bytes()),
        prompt.len()
    );

    let mut gpu = Dsv4Gpu::load(dir, &[0, 1], ActQuantVariant::RefFp8Round, 4096).expect("load");
    gpu.set_prefill_grouped_for_gate(false).expect("arm");
    gpu.set_grouped_route_device_for_gate(false)
        .expect("host routes");

    // The reference is the width that does NOT tile, so the comparison is
    // "tiled against untiled" and not "one tiling against another".
    let reference = run(&gpu, &prompt, TMAX);
    assert_eq!(
        reference.decompositions, 0,
        "width {TMAX} tiled, so it is not an untiled reference and every row below is vacuous"
    );

    for width in [SERVED_WIDTH, 128, 512] {
        let candidate = run(&gpu, &prompt, width);
        assert!(
            candidate.decompositions > 0,
            "width {width} never reached the tiled path, so its match is vacuous"
        );
        // Tiles per decomposition, bounded by the constant rather than asserted
        // from it. A decomposition exists only because its transaction was wider
        // than DSV4_TMAX, so it produces at least 2 tiles, and no dense call in a
        // chunk is wider than the chunk, so it produces at most ceil(width/TMAX).
        // This is the arithmetic that breaks first if the tile width is ever
        // edited apart from DSV4_TMAX again: at a tile of 8 the upper bound is
        // exceeded at every width here.
        let per = candidate.tiles as f64 / candidate.decompositions as f64;
        let ceiling = width.div_ceil(TMAX) as f64;
        assert!(
            (2.0..=ceiling).contains(&per),
            "width {width}: {:.3} tiles per decomposition is outside [2, {ceiling}], which is not \
             what a tile width of DSV4_TMAX={TMAX} produces",
            per
        );
        same_bits(
            &format!("width {width} against untiled width {TMAX}"),
            &reference.digest,
            &candidate.digest,
        )
        .expect("TILING CHANGED THE VALUE: the dense tile width is not a free choice after all");
        println!(
            "EXACT width={width} matches untiled width={TMAX} over logits/live-cache/DSpark/sampled \
             ({} decompositions, {} tiles, {per:.3} per decomposition)",
            candidate.decompositions, candidate.tiles
        );
    }

    // Red arm 1: the comparator itself. One bit of a real digest, and it must
    // refuse and name the byte. A comparator that has never refused is not one.
    let mut perturbed = reference.digest.clone();
    perturbed[7] ^= 0x01;
    let refusal = same_bits("red arm", &reference.digest, &perturbed)
        .expect_err("the bit comparator accepted a digest that differs by one bit");
    assert!(refusal.contains("byte 7"), "{refusal}");
    println!("RED comparator refused a one-bit change: {refusal}");

    // Red arm 2: the digest's sensitivity to the model's own arithmetic. Change
    // ONE token of the prompt and the digest must move, or it is not watching
    // the thing it claims to watch.
    let mut nudged = prompt.clone();
    let last = nudged.len() - 1;
    nudged[last] = if nudged[last] == 0 {
        1
    } else {
        nudged[last] - 1
    };
    assert_ne!(nudged, prompt);
    let nudged_run = run(&gpu, &nudged, SERVED_WIDTH);
    assert!(
        same_bits("red arm 2", &reference.digest, &nudged_run.digest).is_err(),
        "a one-token change to the prompt produced the same digest: the digest is vacuous"
    );
    println!("RED a one-token prompt change moves the digest");

    println!(
        "PASS dense tiling invariant: the tile width is DSV4_TMAX in the code, and every width \
         above it is byte-identical to the untiled walk"
    );
}
