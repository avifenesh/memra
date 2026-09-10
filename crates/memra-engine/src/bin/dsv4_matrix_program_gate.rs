//! Phase consistency of the experimental matrix program, not a quality/perf admission.
use memra_engine::dsv4_gpu::{
    DecodeState, DsparkState, Dsv4Gpu, Dsv4SampleCfg, Dsv4Vt, dsv4_sample_row,
};
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::path::Path;

fn rows_equal(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        assert!(a.is_finite() && b.is_finite());
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "matrix row mismatch at {i}: {a} versus {b}, delta={}",
            (a - b).abs()
        );
    }
}
fn classes(items: Vec<(String, Vec<f32>)>) -> Vec<u8> {
    let mut h = Sha256::new();
    for (name, values) in items {
        h.update(name.as_bytes());
        h.update(values.len().to_le_bytes());
        for v in values {
            h.update(v.to_bits().to_le_bytes());
        }
    }
    h.finalize().to_vec()
}
fn prime(gpu: &Dsv4Gpu, tokens: &[u32], width: usize) -> (Vec<f32>, DecodeState, DsparkState) {
    println!(
        "PRIME matrix={} count={} width={width} device_routes_before={} fresh_alloc_before={}",
        gpu.matrix_moe_enabled(),
        tokens.len(),
        gpu.grouped_device_route_calls(),
        gpu.grouped_fresh_storage_calls()
    );
    let mut state = gpu
        .alloc_decode_state_for_transient(tokens.len() + 128, width.max(gpu.verify_tmax()))
        .expect("state");
    let mut draft = gpu.dspark_alloc_state().expect("draft");
    let row = gpu
        .dspark_prefill_prime_chunked(tokens, &mut state, &mut draft, width)
        .expect("prime");
    (row, state, draft)
}

fn check_frozen_walk(
    gpu: &Dsv4Gpu,
    tokens: &[u32],
    initial: &[f32],
    state: &DecodeState,
    bank: &Path,
) {
    let bytes = std::fs::read(bank).expect("frozen program logits");
    assert_eq!(bytes.len(), 64 * initial.len() * 4, "frozen bank shape");
    let saved = gpu
        .snapshot_decode_state(state)
        .expect("frozen-walk snapshot");
    let mut walk = gpu
        .restore_decode_state_for_transient(&saved, 288, 32)
        .expect("frozen-walk restore");
    let mut row = initial.to_vec();
    for step in 0..64 {
        let offset = step * row.len() * 4;
        let expected: Vec<f32> = bytes[offset..offset + row.len() * 4]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        rows_equal(&row, &expected);
        if step < 63 {
            row = gpu
                .decode_step(tokens[160 + step], &mut walk)
                .expect("frozen forced step");
        }
    }
    println!(
        "EXACT frozen program walk matrix={} rows=64 prefix=160",
        gpu.matrix_moe_enabled()
    );
}

fn main() {
    // Freeze this historical instrument independently of the newer defaults.
    // This is process startup, before any model or worker threads exist.
    unsafe {
        std::env::set_var("MEMRA_DSV4_HC_DOT_SPLIT", "0");
        std::env::set_var("MEMRA_DSV4_DENSE_FAST", "0");
        std::env::set_var("MEMRA_DSV4_NORM_FUSE", "0");
        std::env::set_var("MEMRA_DSV4_NORM_FUSE2", "0");
        std::env::set_var("MEMRA_DSV4_NORM2_WIDE", "0");
        // memra #463 door: the dense wide-prefill tile width. Pinned to the shipped
        // 8 here so an exported 32 cannot silently retile this bin's arm.
        std::env::set_var("MEMRA_DSV4_DENSE_TILE", "8");
        // Gate-only AR phase instrument: pinned off here so no other bin can inherit
        // an exported instrument or null collective from the environment.
        std::env::set_var("MEMRA_DSV4_AR_PHASE", "0");
    }

    let args: Vec<String> = std::env::args().collect();
    assert!(
        (3..=4).contains(&args.len()),
        "usage: dsv4_matrix_program_gate <model-dir> <real-source.txt> [frozen-distribution-dir]"
    );
    let dir = Path::new(&args[1]);
    let source = std::fs::read_to_string(&args[2]).expect("source");
    let tok = Tokenizer::from_hf_dir(dir).expect("tokenizer");
    let tokens = tok.encode(&format!("Review this engine source:\n{source}"), true);
    assert!(tokens.len() > 1200);
    // memra #458: this is a bench process, so it may run the matrix expert program
    // with the default-ON split-K arm; a serving process cannot arm it and refuses
    // that combination at load instead of failing every request.
    memra_engine::arm_matrix_splitk_door_for_gate();
    let mut gpu = Dsv4Gpu::load(dir, &[0, 1], ActQuantVariant::RefFp8Round, 4096).expect("load");
    assert!(
        !gpu.matrix_moe_enabled(),
        "load reference program for the negative controls"
    );
    gpu.set_grouped_route_device_for_gate(true)
        .expect("device routes");
    let (native_row, mut native_state, native_draft) = prime(&gpu, &tokens[..160], 32);
    if let Some(bank) = args.get(3) {
        check_frozen_walk(
            &gpu,
            &tokens,
            &native_row,
            &native_state,
            &Path::new(bank).join("window-0.reference.f32le"),
        );
    }
    let native_host = gpu
        .snapshot_decode_state(&native_state)
        .expect("native snapshot");
    let native_dhost = gpu
        .snapshot_dspark_state(&native_draft)
        .expect("native draft snapshot");
    gpu.set_matrix_moe_for_gate(true).expect("matrix arm");
    assert!(
        gpu.decode_step(tokens[160], &mut native_state)
            .unwrap_err()
            .contains("program mismatch")
    );
    assert!(
        gpu.restore_decode_state(&native_host)
            .err()
            .expect("restore must refuse")
            .contains("program mismatch")
    );
    assert!(
        gpu.restore_dspark_state(&native_dhost)
            .err()
            .expect("draft restore must refuse")
            .contains("program mismatch")
    );
    drop(native_state);
    drop(native_draft);
    drop(native_host);
    drop(native_dhost);
    println!("PASS reference-to-matrix state refusals before execution");

    for (count, widths) in [
        (33, &[1, 32][..]),
        (160, &[1, 32, 64][..]),
        (1025, &[32, 128, 512][..]),
    ] {
        let (baseline, state, draft) = prime(&gpu, &tokens[..count], widths[0]);
        let cache = classes(gpu.cache_classes(&state).expect("cache"));
        let rings = classes(gpu.dspark_ring_classes(&draft).expect("rings"));
        assert!(state.matrix_device_scratch_bytes().iter().all(|&b| b > 0));
        drop(state);
        drop(draft);
        for &width in &widths[1..] {
            let before = gpu.grouped_device_route_calls();
            let (row, state, draft) = prime(&gpu, &tokens[..count], width);
            rows_equal(&baseline, &row);
            assert_eq!(cache, classes(gpu.cache_classes(&state).expect("cache")));
            assert_eq!(
                rings,
                classes(gpu.dspark_ring_classes(&draft).expect("rings"))
            );
            assert!(
                gpu.grouped_device_route_calls() > before,
                "matrix route not engaged"
            );
            println!("EXACT matrix cold count={count} width={width} including position zero");
        }
    }
    let (row, mut state, draft) = prime(&gpu, &tokens[..160], 32);
    if let Some(bank) = args.get(3) {
        check_frozen_walk(
            &gpu,
            &tokens,
            &row,
            &state,
            &Path::new(bank).join("window-0.matrix.f32le"),
        );
    }
    let stored_cache = classes(gpu.cache_classes(&state).expect("persistent cache"));
    let stored_draft = classes(gpu.dspark_ring_classes(&draft).expect("persistent rings"));
    for fresh in [true, false] {
        gpu.set_grouped_fresh_storage_for_gate(fresh).unwrap();
        for width in [1, 32, 64] {
            let before = gpu.grouped_fresh_storage_calls();
            let (other, other_state, other_draft) = prime(&gpu, &tokens[..160], width);
            rows_equal(&row, &other);
            assert_eq!(
                stored_cache,
                classes(gpu.cache_classes(&other_state).unwrap())
            );
            assert_eq!(
                stored_draft,
                classes(gpu.dspark_ring_classes(&other_draft).unwrap())
            );
            let calls = gpu.grouped_fresh_storage_calls() - before;
            assert_eq!(calls > 0, fresh, "fresh-storage control engagement");
            println!("EXACT matrix storage fresh={fresh} width={width} calls={calls}");
        }
    }
    let delta = native_row
        .iter()
        .zip(&row)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    println!(
        "CHARACTERIZE reference/matrix max_logit_delta={delta}; no quality threshold inferred"
    );
    let host = gpu.snapshot_decode_state(&state).expect("matrix snapshot");
    let dhost = gpu
        .snapshot_dspark_state(&draft)
        .expect("matrix draft snapshot");
    for t in [1, 2, 6] {
        let mut a = gpu
            .restore_decode_state_for_transient(&host, 288, 32)
            .expect("plain restore");
        let mut b = gpu
            .restore_decode_state_for_transient(&host, 288, 32)
            .expect("verify restore");
        let mut expected = Vec::new();
        for &input in &tokens[160..160 + t] {
            expected.extend(gpu.decode_step(input, &mut a).expect("plain forced step"));
        }
        let mut verify = gpu.alloc_verify_state_for(288).expect("verify");
        let (actual, _) = gpu
            .verify_batch_dev(&tokens[160..160 + t], &mut b, &mut verify, None, true)
            .expect("forced verify");
        rows_equal(&expected, &actual.expect("full rows"));
        gpu.commit_verify_dev(&mut b, &mut verify, t)
            .expect("commit full");
        assert_eq!(
            classes(gpu.cache_classes(&a).unwrap()),
            classes(gpu.cache_classes(&b).unwrap())
        );
        println!("EXACT matrix plain/verify t={t} logits and committed state");
        if t > 1 {
            let mut a = gpu
                .restore_decode_state_for_transient(&host, 288, 32)
                .unwrap();
            let mut b = gpu
                .restore_decode_state_for_transient(&host, 288, 32)
                .unwrap();
            gpu.decode_step(tokens[160], &mut a).unwrap();
            gpu.verify_batch_dev(&tokens[160..160 + t], &mut b, &mut verify, None, true)
                .unwrap();
            gpu.commit_verify_dev(&mut b, &mut verify, 1).unwrap();
            assert_eq!(
                classes(gpu.cache_classes(&a).unwrap()),
                classes(gpu.cache_classes(&b).unwrap())
            );
            rows_equal(
                &gpu.decode_step(tokens[161], &mut a).unwrap(),
                &gpu.decode_step(tokens[161], &mut b).unwrap(),
            );
            println!("EXACT matrix rollback t={t} accepted=1 and next-step logits");
        }
    }
    let cfg = Dsv4SampleCfg {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        seed: 20260906,
    };
    let mut a = gpu
        .restore_decode_state_for_transient(&host, 288, 32)
        .unwrap();
    let mut expected = Vec::new();
    let mut current = row.clone();
    for p in 0..32 {
        let id = dsv4_sample_row(&current, 160 + p, &cfg).unwrap();
        expected.push(id);
        if p < 31 {
            current = gpu.decode_step(id, &mut a).unwrap();
        }
    }
    let mut b = gpu
        .restore_decode_state_for_transient(&host, 288, 32)
        .unwrap();
    let mut ds = gpu.restore_dspark_state(&dhost).unwrap();
    let mut verify = gpu.alloc_verify_state_for(288).unwrap();
    let sampled = gpu
        .spec_sampled_batched_pen_restored(
            &tokens[..160],
            &row,
            32,
            &mut b,
            &mut ds,
            &mut verify,
            usize::MAX,
            Dsv4Vt::Off,
            &cfg,
            None,
            None,
        )
        .expect("sampled spec");
    assert_eq!(expected, sampled.tokens);
    assert!(!sampled.rounds.is_empty());
    println!(
        "EXACT matrix sampled plain/DSpark tokens=32 rounds={}",
        sampled.rounds.len()
    );
    gpu.set_matrix_moe_for_gate(false).unwrap();
    assert!(
        gpu.decode_step(tokens[160], &mut state)
            .unwrap_err()
            .contains("program mismatch")
    );
    assert!(
        gpu.restore_decode_state(&host)
            .err()
            .unwrap()
            .contains("program mismatch")
    );
    assert!(
        gpu.restore_dspark_state(&dhost)
            .err()
            .unwrap()
            .contains("program mismatch")
    );
    println!(
        "PASS matrix-to-reference state refusals; matrix phase gates complete, checkpoint quality/performance/serving still unqualified"
    );
}
