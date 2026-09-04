//! Lossless DSV4 pinned-host park/restore gate.
//!
//! One model load, two independent arms:
//!   * plain trunk state: snapshot -> capacity-growing restore -> identical logits/state;
//!   * trunk + DSpark state: snapshot -> restore -> identical proposal, logits, trunk
//!     cache classes and persistent drafter rings.
//!
//! Device-to-device equality is the correct instrument here: both arms use the same
//! numeric realization and differ only by a D2H/H2D state round trip. Every live f32
//! element is compared by bits. Dead capacity tails and scratch are intentionally absent.
//!
//! Usage: dsv4-host-cache-gate <model-dir> <fixtures.json> [dev0,dev1]

use memra_engine::dsv4_gpu::{DecodeState, DsparkState, Dsv4Gpu};
use memra_gguf::dsv4_forward::{FixtureSpec, drift_coeff};
use std::path::Path;

fn argmax(row: &[f32]) -> u32 {
    let mut best = 0usize;
    for i in 1..row.len() {
        if row[i] > row[best] {
            best = i;
        }
    }
    best as u32
}

fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn max_abs(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

/// Existing DSV4 native-class doctrine, specialized to two GPU realizations: each
/// side contributes the lane-7 depth-86 drift coefficient.
fn native_gpu_pair_band(top1: f32) -> f64 {
    let c = drift_coeff(86.0, 86.0);
    3.0 * 2f64.sqrt() * (c + c) * top1.abs() as f64
}

fn classes_equal(left: &[(String, Vec<f32>)], right: &[(String, Vec<f32>)]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|((ln, lv), (rn, rv))| ln == rn && bits_equal(lv, rv))
}

fn plain_warm(gpu: &Dsv4Gpu, prompt: &[u32], capacity: usize, warm: usize) -> (DecodeState, u32) {
    let mut state = gpu
        .alloc_decode_state_for(capacity)
        .expect("plain capacity-planned state");
    let pre = gpu
        .prefill_with_cache(prompt, &mut state)
        .expect("plain prefill");
    let mut token = argmax(&pre.logits);
    for _ in 0..warm {
        token = gpu
            .decode_step_greedy(token, &mut state)
            .expect("plain warm step");
    }
    (state, token)
}

fn dspark_warm(
    gpu: &Dsv4Gpu,
    prompt: &[u32],
    capacity: usize,
    warm: usize,
) -> (DecodeState, DsparkState, u32) {
    let mut state = gpu
        .alloc_decode_state_for(capacity)
        .expect("DSpark capacity-planned state");
    let mut dstate = gpu.dspark_alloc_state().expect("DSpark state");
    let pre = gpu
        .dspark_prefill_prime(prompt, &mut state, &mut dstate)
        .expect("DSpark prefill");
    let mut token = argmax(&pre.logits);
    for _ in 0..warm {
        let pos = state.pos;
        token = gpu
            .decode_step_greedy_tap(token, &mut state, &mut dstate, 0)
            .expect("DSpark warm step");
        gpu.dspark_write_rings(&mut dstate, 0, pos)
            .expect("DSpark warm ring write");
    }
    (state, dstate, token)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: dsv4-host-cache-gate <model-dir> <fixtures.json> [dev0,dev1]");
        std::process::exit(2);
    }
    let model_dir = Path::new(&args[1]);
    let fixture = FixtureSpec::load(Path::new(&args[2]));
    let devices: Vec<usize> = args
        .get(3)
        .map(|raw| {
            raw.split(',')
                .map(|part| part.parse().expect("device index"))
                .collect()
        })
        .unwrap_or_else(|| vec![0, 1]);
    let prompt = &fixture
        .tokens_160
        .as_ref()
        .expect("fixture tokens_160 required")[..66];
    let warm = 17usize; // crosses several CSA phases without making the gate slow
    let continuation = 11usize;
    let small_capacity = prompt.len() + warm + continuation + 1;
    let grown_capacity = small_capacity + 97;
    let gpu = Dsv4Gpu::load(model_dir, &devices, fixture.variant, grown_capacity)
        .expect("load DSV4 GPU model");
    assert!(
        gpu.dspark.is_some(),
        "gate requires MEMRA_DSV4_DRAFTER=dspark"
    );

    // Bounded-prefill transaction widths must not change the realized trunk or drafter
    // state. Width 64 crosses both the shipped speculative ceiling (6) and the largest
    // register-specialized twin (32), exercising the tiled exact-kernel fallback and the
    // advertised prefill maximum, so this catches fixed-T assumptions instead of merely
    // re-running the DSpark shape.
    let chunk_width = 64usize;
    let mut chunk_plain_1 = gpu
        .alloc_decode_state_for_transient(small_capacity, chunk_width)
        .expect("chunk plain width-1 state");
    let mut chunk_plain_17 = gpu
        .alloc_decode_state_for_transient(small_capacity, chunk_width)
        .expect("chunk plain width-17 state");
    let logits_plain_1 = gpu
        .prefill_with_cache_chunked(prompt, &mut chunk_plain_1, 1)
        .expect("chunk plain width 1");
    let logits_plain_17 = gpu
        .prefill_with_cache_chunked(prompt, &mut chunk_plain_17, chunk_width)
        .expect("chunk plain width 17");
    assert!(bits_equal(&logits_plain_1, &logits_plain_17));
    assert!(classes_equal(
        &gpu.cache_classes(&chunk_plain_1)
            .expect("chunk plain classes width 1"),
        &gpu.cache_classes(&chunk_plain_17)
            .expect("chunk plain classes width 17"),
    ));
    let mut monolithic_plain = gpu
        .alloc_decode_state_for_transient(small_capacity, chunk_width)
        .expect("monolithic semantic state");
    let monolithic_logits = gpu
        .prefill_with_cache(prompt, &mut monolithic_plain)
        .expect("monolithic semantic prefill")
        .logits;
    let logits_maxabs = max_abs(&monolithic_logits, &logits_plain_1);
    let monolithic_cache_bits = classes_equal(
        &gpu.cache_classes(&monolithic_plain)
            .expect("monolithic semantic cache classes"),
        &gpu.cache_classes(&chunk_plain_1)
            .expect("chunk semantic cache classes"),
    );
    let mut monolithic_row = monolithic_logits;
    let mut chunk_row = logits_plain_1.clone();
    let mut semantic_agree = 0usize;
    let mut semantic_in_band = 0usize;
    let mut semantic_out_of_band = 0usize;
    let mut semantic_maxabs = 0.0f32;
    for step in 0..16 {
        semantic_maxabs = semantic_maxabs.max(max_abs(&monolithic_row, &chunk_row));
        let monolithic_token = argmax(&monolithic_row);
        let chunk_token = argmax(&chunk_row);
        if monolithic_token == chunk_token {
            semantic_agree += 1;
        } else {
            let margin = (monolithic_row[monolithic_token as usize]
                - monolithic_row[chunk_token as usize]) as f64;
            let band = native_gpu_pair_band(monolithic_row[monolithic_token as usize]);
            if margin <= band {
                semantic_in_band += 1;
            } else {
                semantic_out_of_band += 1;
            }
            println!(
                "[dsv4-chunk-semantics] disagreement step={step} mono={monolithic_token} chunk={chunk_token} margin={margin:.6} band={band:.6} class={}",
                if margin <= band {
                    "IN-BAND"
                } else {
                    "OUT-OF-BAND"
                }
            );
        }
        if step + 1 < 16 {
            // Teacher-force the monolithic pick into BOTH states. A free-running
            // comparison turns one legitimate near-tie into an unrelated tail.
            monolithic_row = gpu
                .decode_step(monolithic_token, &mut monolithic_plain)
                .expect("monolithic semantic teacher force");
            chunk_row = gpu
                .decode_step(monolithic_token, &mut chunk_plain_1)
                .expect("chunk semantic teacher force");
        }
    }
    assert_eq!(
        semantic_out_of_band, 0,
        "monolithic/chunk teacher forcing has out-of-band disagreements"
    );
    println!(
        "[dsv4-chunk-semantics] monolithic_vs_chunk1 initial_logits_maxabs={logits_maxabs:.8} stream_maxabs={semantic_maxabs:.8} cache_bits_equal={monolithic_cache_bits} teacher_forced16={semantic_agree}_agree,{semantic_in_band}_in_band,{semantic_out_of_band}_out_of_band"
    );

    let mut chunk_spec_1 = gpu
        .alloc_decode_state_for_transient(small_capacity, chunk_width)
        .expect("chunk DSpark width-1 state");
    let mut chunk_spec_17 = gpu
        .alloc_decode_state_for_transient(small_capacity, chunk_width)
        .expect("chunk DSpark width-17 state");
    let mut chunk_ds_1 = gpu
        .dspark_alloc_state()
        .expect("chunk DSpark width-1 rings");
    let mut chunk_ds_17 = gpu
        .dspark_alloc_state()
        .expect("chunk DSpark width-17 rings");
    let logits_spec_1 = gpu
        .dspark_prefill_prime_chunked(prompt, &mut chunk_spec_1, &mut chunk_ds_1, 1)
        .expect("chunk DSpark width 1");
    let logits_spec_17 = gpu
        .dspark_prefill_prime_chunked(prompt, &mut chunk_spec_17, &mut chunk_ds_17, chunk_width)
        .expect("chunk DSpark width 17");
    assert!(bits_equal(&logits_spec_1, &logits_spec_17));
    assert!(classes_equal(
        &gpu.cache_classes(&chunk_spec_1)
            .expect("chunk DSpark trunk classes width 1"),
        &gpu.cache_classes(&chunk_spec_17)
            .expect("chunk DSpark trunk classes width 17"),
    ));
    assert!(classes_equal(
        &gpu.dspark_ring_classes(&chunk_ds_1)
            .expect("chunk DSpark rings width 1"),
        &gpu.dspark_ring_classes(&chunk_ds_17)
            .expect("chunk DSpark rings width 17"),
    ));
    let chunk_token = argmax(&logits_spec_1);
    let chunk_tap_1 = chunk_ds_1.tap_head;
    let chunk_tap_17 = chunk_ds_17.tap_head;
    let chunk_prop_1 = gpu
        .dspark_forward_spec(
            &mut chunk_ds_1,
            chunk_token,
            chunk_tap_1,
            chunk_spec_1.pos - 1,
            false,
        )
        .expect("chunk DSpark proposal width 1");
    let chunk_prop_17 = gpu
        .dspark_forward_spec(
            &mut chunk_ds_17,
            chunk_token,
            chunk_tap_17,
            chunk_spec_17.pos - 1,
            false,
        )
        .expect("chunk DSpark proposal width 17");
    assert_eq!(chunk_prop_1.out_ids, chunk_prop_17.out_ids);
    assert!(bits_equal(
        &chunk_prop_1.confidence,
        &chunk_prop_17.confidence
    ));
    let mut monolithic_spec = gpu
        .alloc_decode_state_for_transient(small_capacity, chunk_width)
        .expect("monolithic DSpark semantic state");
    let mut monolithic_ds = gpu
        .dspark_alloc_state()
        .expect("monolithic DSpark semantic rings");
    let monolithic_spec_logits = gpu
        .dspark_prefill_prime(prompt, &mut monolithic_spec, &mut monolithic_ds)
        .expect("monolithic DSpark semantic prefill")
        .logits;
    let monolithic_spec_token = argmax(&monolithic_spec_logits);
    assert_eq!(monolithic_spec_token, chunk_token);
    let monolithic_tap = monolithic_ds.tap_head;
    let monolithic_prop = gpu
        .dspark_forward_spec(
            &mut monolithic_ds,
            monolithic_spec_token,
            monolithic_tap,
            monolithic_spec.pos - 1,
            false,
        )
        .expect("monolithic DSpark semantic proposal");
    println!(
        "[dsv4-chunk-semantics] monolithic_vs_chunk_dspark proposal_ids_equal={} confidence_bits_equal={}",
        monolithic_prop.out_ids == chunk_prop_1.out_ids,
        bits_equal(&monolithic_prop.confidence, &chunk_prop_1.confidence),
    );

    // Plain trunk round trip, including restore into a larger capacity allocation.
    let (mut plain_a, mut plain_token) = plain_warm(&gpu, prompt, small_capacity, warm);
    let plain_host = gpu
        .snapshot_decode_state(&plain_a)
        .expect("snapshot plain state");
    let plain_host_bytes = plain_host.bytes();
    let mut plain_b = gpu
        .restore_decode_state_for(&plain_host, grown_capacity)
        .expect("restore grown plain state");
    assert_eq!(plain_b.capacity, grown_capacity);
    assert!(classes_equal(
        &gpu.cache_classes(&plain_a).expect("plain classes A"),
        &gpu.cache_classes(&plain_b).expect("plain classes B"),
    ));
    for _ in 0..continuation {
        let row_a = gpu.decode_step(plain_token, &mut plain_a).expect("plain A");
        let row_b = gpu.decode_step(plain_token, &mut plain_b).expect("plain B");
        assert!(bits_equal(&row_a, &row_b), "plain restored logits differ");
        plain_token = argmax(&row_a);
    }
    assert!(classes_equal(
        &gpu.cache_classes(&plain_a).expect("plain final classes A"),
        &gpu.cache_classes(&plain_b).expect("plain final classes B"),
    ));

    // Trunk + bundled DSpark state round trip. Proposal equality exercises the restored
    // newest-tap row; ring equality exercises every persistent drafter row.
    let (mut spec_a, mut ds_a, spec_token) = dspark_warm(&gpu, prompt, small_capacity, warm);
    let spec_host = gpu
        .snapshot_decode_state(&spec_a)
        .expect("snapshot DSpark trunk");
    let ds_host = gpu
        .snapshot_dspark_state(&ds_a)
        .expect("snapshot DSpark state");
    let spec_host_bytes = spec_host.bytes() + ds_host.bytes();
    let mut spec_b = gpu
        .restore_decode_state_for(&spec_host, grown_capacity)
        .expect("restore DSpark trunk");
    let mut ds_b = gpu
        .restore_dspark_state(&ds_host)
        .expect("restore DSpark state");

    assert!(classes_equal(
        &gpu.cache_classes(&spec_a).expect("DSpark trunk classes A"),
        &gpu.cache_classes(&spec_b).expect("DSpark trunk classes B"),
    ));
    assert!(classes_equal(
        &gpu.dspark_ring_classes(&ds_a).expect("DSpark rings A"),
        &gpu.dspark_ring_classes(&ds_b).expect("DSpark rings B"),
    ));
    let tap_a = ds_a.tap_head;
    let tap_b = ds_b.tap_head;
    let prop_a = gpu
        .dspark_forward_spec(&mut ds_a, spec_token, tap_a, spec_a.pos - 1, false)
        .expect("DSpark proposal A");
    let prop_b = gpu
        .dspark_forward_spec(&mut ds_b, spec_token, tap_b, spec_b.pos - 1, false)
        .expect("DSpark proposal B");
    assert_eq!(prop_a.out_ids, prop_b.out_ids, "restored DSpark ids differ");
    assert!(bits_equal(&prop_a.confidence, &prop_b.confidence));

    let pos_a = spec_a.pos;
    let pos_b = spec_b.pos;
    let row_a = gpu
        .decode_step_tap(spec_token, &mut spec_a, &mut ds_a, 0)
        .expect("DSpark continuation A");
    let row_b = gpu
        .decode_step_tap(spec_token, &mut spec_b, &mut ds_b, 0)
        .expect("DSpark continuation B");
    assert!(bits_equal(&row_a, &row_b), "restored DSpark logits differ");
    gpu.dspark_write_rings(&mut ds_a, 0, pos_a)
        .expect("DSpark ring A");
    gpu.dspark_write_rings(&mut ds_b, 0, pos_b)
        .expect("DSpark ring B");
    assert!(classes_equal(
        &gpu.cache_classes(&spec_a).expect("DSpark final trunk A"),
        &gpu.cache_classes(&spec_b).expect("DSpark final trunk B"),
    ));
    assert!(classes_equal(
        &gpu.dspark_ring_classes(&ds_a)
            .expect("DSpark final rings A"),
        &gpu.dspark_ring_classes(&ds_b)
            .expect("DSpark final rings B"),
    ));

    println!(
        "[dsv4-host-cache-gate] PASS prompt={} chunk_widths=1,{} warm={} continuation={} capacity={}=>{} plain_host_bytes={} dspark_host_bytes={}",
        prompt.len(),
        chunk_width,
        warm,
        continuation,
        small_capacity,
        grown_capacity,
        plain_host_bytes,
        spec_host_bytes,
    );
}
