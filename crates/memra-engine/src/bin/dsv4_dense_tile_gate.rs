//! Dense wide-prefill tile width (memra #463): the numeric-class gate and the
//! per-arm cost, on one real prompt at the SERVED chunk width.
//!
//! The claim under test. Above `DSV4_TMAX` the dense entry points decompose a
//! wide transaction into tiles of 8 rows and relaunch. Tiling at 32 instead is a
//! 4x launch reduction at the served width of 64 with no new kernel, and it is
//! argued to be BIT-EQUAL because M only decides how many independent
//! accumulators a block keeps in registers, never the order of one accumulator's
//! adds. This binary refuses to take that argument on trust: it hashes logits,
//! every live cache class, the DSpark rings and a full sampled speculative
//! continuation for each arm, and compares them byte for byte, naming the first
//! differing byte when they disagree.
//!
//! What makes it a check rather than a ritual:
//! - caller: `main`, every run, both arms, no flag needed to reach either;
//! - non-vacuous input: a real >= 1025-token source prompt (padding refused),
//!   with the digest asserted to be sensitive to a ONE-TOKEN change in it;
//! - engagement: the per-arm tiled-launch counters must be non-zero and must
//!   stand in the 4:1 ratio the widths predict, so an arm that changed nothing
//!   cannot pass as an arm;
//! - red arm: two, both run every time. The comparator is handed a one-ULP
//!   perturbation of a real digest and must refuse naming the byte, and the
//!   whole pipeline is re-run on a perturbed prompt whose digest must differ.
use memra_engine::dsv4_gpu::{
    Dsv4Gpu, Dsv4SampleCfg, Dsv4Vt, dense_tile_counts_for_gate, dense_tile_width_for_gate,
    set_dense_tile_width_for_gate,
};
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::path::Path;
use std::time::Instant;

/// The width a customer request prefills at today (`DSV4_SERVING_BATCH_WIDTH_MAX`
/// is 64 and chunked prefill runs at it). Measuring the door anywhere else would
/// measure a transaction nobody serves.
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

/// Refuse on the FIRST differing bit, and say where it is. A bare `assert_eq!`
/// on two hashes tells the reader that something moved and nothing else.
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

struct Arm {
    digest: Vec<u8>,
    prefill_seconds: f64,
    tiled_launches: u64,
}

fn run(gpu: &Dsv4Gpu, prompt: &[u32], width: i32) -> Arm {
    // Selection is host-thread state read at enqueue; drain first so no in-flight
    // launch straddles the switch, exactly as the dense exact-tail seam requires.
    for stage in &gpu.stages {
        stage.gpu.stream().synchronize().expect("drain before arm");
    }
    set_dense_tile_width_for_gate(width).expect("arm the tile width");
    assert_eq!(dense_tile_width_for_gate(), width, "arm identity");
    let before = dense_tile_counts_for_gate();

    let mut state = gpu
        .alloc_decode_state_for_transient(prompt.len() + 32, SERVED_WIDTH)
        .expect("cache");
    let mut draft = gpu.dspark_alloc_state().expect("draft");
    let timer = Instant::now();
    let logits = gpu
        .dspark_prefill_prime_chunked(prompt, &mut state, &mut draft, SERVED_WIDTH)
        .expect("served-width prime");
    let prefill_seconds = timer.elapsed().as_secs_f64();

    let mut hash = Sha256::new();
    floats(&mut hash, &logits, false);
    classes(&mut hash, gpu.cache_classes(&state).expect("trunk classes"));
    classes(
        &mut hash,
        gpu.dspark_ring_classes(&draft).expect("draft classes"),
    );
    // Vendor-default sampled shape is what serves, so the class question is asked
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
    let slot = usize::from(width == 32);
    let other = 1 - slot;
    assert_eq!(
        after[other], before[other],
        "the other arm's counter moved while width={width} was armed"
    );
    let tiled_launches = after[slot] - before[slot];
    assert!(
        tiled_launches > 0,
        "width={width} tiled nothing: the transaction never got wide enough to reach the door"
    );
    println!(
        "ARM tile={width} width={SERVED_WIDTH} tokens={} prefill_seconds={prefill_seconds:.6} \
         tiled_launches={tiled_launches}",
        prompt.len()
    );
    Arm {
        digest: hash.finalize().to_vec(),
        prefill_seconds,
        tiled_launches,
    }
}

fn main() {
    // Freeze the other dense-family doors so this gate measures ONE difference.
    // Process startup, before any model or worker thread exists.
    unsafe {
        std::env::set_var("MEMRA_DSV4_HC_DOT_SPLIT", "0");
        std::env::set_var("MEMRA_DSV4_DENSE_FAST", "0");
        std::env::set_var("MEMRA_DSV4_NORM_FUSE", "0");
        std::env::set_var("MEMRA_DSV4_NORM_FUSE2", "0");
        std::env::set_var("MEMRA_DSV4_NORM2_WIDE", "0");
        std::env::set_var("MEMRA_DSV4_AR_PHASE", "0");
        // The door itself is armed through the gate seam per arm, never inherited.
        std::env::remove_var("MEMRA_DSV4_DENSE_TILE");
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
    assert!(
        prompt.len() > SERVED_WIDTH,
        "a prompt no wider than one chunk never reaches the tiled path"
    );
    println!(
        "SOURCE sha256={:x} tokens={}",
        Sha256::digest(source.as_bytes()),
        prompt.len()
    );

    let mut gpu = Dsv4Gpu::load(dir, &[0, 1], ActQuantVariant::RefFp8Round, 4096).expect("load");
    gpu.set_prefill_grouped_for_gate(false).expect("arm");
    gpu.set_grouped_route_device_for_gate(false)
        .expect("host routes");

    // The shipped arm first, then the door, then the shipped arm again: a drift
    // that swallowed the difference would have to swallow it in both directions.
    let control = run(&gpu, &prompt, 8);
    let candidate = run(&gpu, &prompt, 32);
    let control_again = run(&gpu, &prompt, 8);

    same_bits(
        "tile 8 against itself",
        &control.digest,
        &control_again.digest,
    )
    .expect("the shipped arm is not reproducible on this box, so nothing here is measurable");
    same_bits("tile 32 against tile 8", &control.digest, &candidate.digest)
        .expect("NEW NUMERIC CLASS: the tile width moved a bit, so this door owes drift rows");
    println!("EXACT tile=32 matches tile=8 over logits/live-cache/DSpark/sampled");

    // Engagement, in the shape the widths predict. 1025 tokens at width 64 is 17
    // chunks, 16 of them wide enough to tile; the ratio is what proves the door
    // did the thing it claims and not merely something.
    let ratio = control.tiled_launches as f64 / candidate.tiled_launches as f64;
    println!(
        "LAUNCHES tile8={} tile32={} ratio={ratio:.4}",
        control.tiled_launches, candidate.tiled_launches
    );
    assert!(
        (3.5..=4.5).contains(&ratio),
        "tile 8 should issue about 4x the tiled decompositions of tile 32, got {ratio:.4}"
    );

    // Red arm 1: the comparator itself. One bit of a real digest, and it must
    // refuse and name the byte. A comparator that has never refused is not one.
    let mut perturbed = candidate.digest.clone();
    perturbed[7] ^= 0x01;
    let refusal = same_bits("red arm", &candidate.digest, &perturbed)
        .expect_err("the bit comparator accepted a digest that differs by one bit");
    assert!(refusal.contains("byte 7"), "{refusal}");
    println!("RED comparator refused a one-bit change: {refusal}");

    // Red arm 2: the digest's sensitivity to the model's own arithmetic. Change
    // ONE token of the prompt and the whole pipeline must produce a different
    // digest, or the digest is not watching the thing it claims to watch.
    let mut nudged = prompt.clone();
    let last = nudged.len() - 1;
    nudged[last] = if nudged[last] == 0 {
        1
    } else {
        nudged[last] - 1
    };
    assert_ne!(nudged, prompt);
    let nudged_arm = run(&gpu, &nudged, 8);
    assert!(
        same_bits("red arm 2", &control.digest, &nudged_arm.digest).is_err(),
        "a one-token change to the prompt produced the same digest: the digest is vacuous"
    );
    println!("RED a one-token prompt change moves the digest");

    // Cost, reported and never asserted on: this box shares its cards, and a
    // serving decision needs the served-path rows in the lane doc, not a number
    // a gate happened to see between two other lanes' cells.
    println!(
        "COST tile8={:.6}s tile8_repeat={:.6}s tile32={:.6}s speedup={:.4}x",
        control.prefill_seconds,
        control_again.prefill_seconds,
        candidate.prefill_seconds,
        control.prefill_seconds / candidate.prefill_seconds
    );
    println!(
        "PASS dense wide tile: SAME NUMERIC CLASS at the served width; serving qualification \
         remains separate"
    );
}
