//! Fresh/direct-restored active host C4 versus the canonical all-device program.
use memra_engine::dsv4_gpu::{
    DecodeState, DsparkState, Dsv4Gpu, Dsv4SampleCfg, Dsv4Vt, dsv4_sample_row,
};
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::path::Path;

fn rows_equal(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert!(a.is_finite() && b.is_finite());
        assert_eq!(a.to_bits(), b.to_bits(), "logit mismatch at {i}");
    }
}
fn classes(items: Vec<(String, Vec<f32>)>) -> Vec<u8> {
    let mut h = Sha256::new();
    for (name, values) in items {
        h.update(name.as_bytes());
        h.update((values.len() as u64).to_le_bytes());
        for value in values {
            h.update(value.to_le_bytes());
        }
    }
    h.finalize().to_vec()
}
fn continuation(
    gpu: &Dsv4Gpu,
    tokens: &[u32],
    row: &[f32],
    mut state: DecodeState,
    mut draft: DsparkState,
    count: usize,
    width: usize,
) -> Vec<u8> {
    let suffix = 129;
    let mut row = if suffix > 0 {
        gpu.dspark_continue_prefix_chunked(
            &tokens[count..count + suffix],
            &mut state,
            &mut draft,
            width,
        )
        .expect("suffix")
    } else {
        row.to_vec()
    };
    let cfg = Dsv4SampleCfg {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        seed: 20260906,
    };
    let mut prompt = tokens[..count + suffix].to_vec();
    let mut hash = Sha256::new();
    for _ in 0..9 {
        let token = dsv4_sample_row(&row, state.pos, &cfg).expect("sample");
        prompt.push(token);
        let pos = state.pos;
        row = gpu
            .decode_step_tap(token, &mut state, &mut draft, 0)
            .expect("plain");
        gpu.dspark_write_rings(&mut draft, 0, pos).expect("rings");
        hash.update(token.to_le_bytes());
        for value in &row {
            hash.update(value.to_le_bytes());
        }
    }
    let mut verify = gpu.alloc_verify_state_for(state.capacity).expect("verify");
    let sampled = gpu
        .spec_sampled_batched_pen_restored(
            &prompt,
            &row,
            32,
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
    assert_eq!(sampled.tokens.len(), 32);
    assert!(!sampled.rounds.is_empty());
    for token in sampled.tokens {
        hash.update(token.to_le_bytes());
    }
    for round in sampled.rounds {
        for n in [
            round.start_pos,
            round.accepts,
            round.verified,
            round.t_batch,
            round.emitted,
        ] {
            hash.update((n as u64).to_le_bytes());
        }
        for value in round.confidence {
            hash.update(value.to_le_bytes());
        }
    }
    hash.update(classes(gpu.cache_classes(&state).unwrap()));
    hash.update(classes(gpu.dspark_ring_classes(&draft).unwrap()));
    hash.finalize().to_vec()
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
    }

    let args: Vec<String> = std::env::args().collect();
    assert!(
        (3..=4).contains(&args.len())
            && args
                .get(3)
                .is_none_or(|s| matches!(s.as_str(), "capacity-1m" | "recent-512")),
        "usage: dsv4_c4_direct_gate <model-dir> <source.txt> [capacity-1m|recent-512]"
    );
    let dir = Path::new(&args[1]);
    let source = std::fs::read_to_string(&args[2]).expect("source");
    let tokenizer = Tokenizer::from_hf_dir(dir).expect("tokenizer");
    let tokens = tokenizer.encode(&format!("Review this inference code:\n{source}"), true);
    assert!(tokens.len() > 5000);
    println!("SOURCE sha256={:x}", Sha256::digest(source.as_bytes()));
    let capacity_test = args.get(3).is_some_and(|arg| arg == "capacity-1m");
    let recent_rows = if args.get(3).is_some_and(|arg| arg == "recent-512") {
        512
    } else {
        0
    };
    let max_seq = if capacity_test { 1_048_576 } else { 8192 };
    let gpu = Dsv4Gpu::load(dir, &[0, 1], ActQuantVariant::RefFp8Round, max_seq).expect("model");
    assert!(
        gpu.matrix_moe_enabled(),
        "fresh host C4 needs the matrix program"
    );
    for (count, width) in [(1, 32), (160, 32), (1025, 512), (4097, 512)] {
        let capacity = count + 256;
        let mut reference = gpu
            .alloc_decode_state_for_transient(capacity, width)
            .unwrap();
        let reference_bytes = reference.cache_bytes.clone();
        let mut reference_draft = gpu.dspark_alloc_state().unwrap();
        let reference_row = gpu
            .dspark_prefill_prime_chunked(
                &tokens[..count],
                &mut reference,
                &mut reference_draft,
                width,
            )
            .unwrap();
        let reference_cache = classes(gpu.cache_classes(&reference).unwrap());
        let reference_rings = classes(gpu.dspark_ring_classes(&reference_draft).unwrap());
        let reference_snapshot = gpu.snapshot_decode_state(&reference).unwrap();
        let draft_snapshot = gpu.snapshot_dspark_state(&reference_draft).unwrap();
        let expected = continuation(
            &gpu,
            &tokens,
            &reference_row,
            reference,
            reference_draft,
            count,
            width,
        );
        let mut direct = gpu
            .alloc_decode_state_host_c4_recent(capacity, width, recent_rows)
            .expect("direct host allocation");
        assert_eq!(
            direct.host_cache_bytes,
            gpu.c4_host_bytes_for_capacity(capacity).unwrap()
        );
        assert!(direct.host_cache_bytes.iter().sum::<u64>() > 0);
        let recent_bytes = gpu.c4_recent_gpu_bytes(&direct).unwrap();
        assert_eq!(recent_bytes.iter().sum::<u64>() > 0, recent_rows > 0);
        for (stage, &expected_bytes) in reference_bytes.iter().enumerate() {
            assert_eq!(
                direct.cache_bytes[stage] + direct.host_cache_bytes[stage] - recent_bytes[stage],
                expected_bytes
            );
        }
        println!(
            "ALLOCATION count={count} width={width} gpu_bytes={:?} host_bytes={:?} recent_rows={recent_rows} recent_gpu_bytes={recent_bytes:?} no_initial_C4_gpu_history=true",
            direct.cache_bytes, direct.host_cache_bytes,
        );
        let mut direct_draft = gpu.dspark_alloc_state().unwrap();
        let direct_row = gpu
            .dspark_prefill_prime_chunked(&tokens[..count], &mut direct, &mut direct_draft, width)
            .expect("fresh host prime including pos0");
        rows_equal(&reference_row, &direct_row);
        assert_eq!(
            reference_cache,
            classes(gpu.cache_classes(&direct).unwrap())
        );
        assert_eq!(
            reference_rings,
            classes(gpu.dspark_ring_classes(&direct_draft).unwrap())
        );
        let direct_snapshot = gpu.snapshot_decode_state(&direct).unwrap();
        assert_eq!(reference_snapshot.bytes(), direct_snapshot.bytes());
        assert_eq!(
            expected,
            continuation(
                &gpu,
                &tokens,
                &direct_row,
                direct,
                direct_draft,
                count,
                width
            )
        );
        println!(
            "EXACT fresh host/device count={count} width={width} pos0, cache, rings, suffix, sampled plain/spec"
        );
        for resize in [capacity, capacity + 128, count + 200] {
            for (name, snapshot, host) in [
                ("device-to-host", &reference_snapshot, true),
                ("host-to-device", &direct_snapshot, false),
                ("host-to-host", &direct_snapshot, true),
            ] {
                let state = if host {
                    gpu.restore_decode_state_host_c4_recent(snapshot, resize, width, recent_rows)
                        .expect("direct host restore")
                } else {
                    gpu.restore_decode_state_for_transient(snapshot, resize, width)
                        .expect("device restore")
                };
                assert_eq!(state.pos, count);
                assert_eq!(reference_cache, classes(gpu.cache_classes(&state).unwrap()));
                assert_eq!(state.host_cache_bytes.iter().sum::<u64>() > 0, host);
                assert_eq!(
                    gpu.c4_recent_gpu_bytes(&state).unwrap().iter().sum::<u64>() > 0,
                    host && recent_rows > 0
                );
                let draft = gpu.restore_dspark_state(&draft_snapshot).unwrap();
                assert_eq!(
                    expected,
                    continuation(&gpu, &tokens, &reference_row, state, draft, count, width),
                    "restore {name} capacity={resize}"
                );
                println!("EXACT restore={name} count={count} capacity={resize} width={width}");
            }
        }
    }
    if recent_rows > 0 {
        assert!(gpu.c4_recent_gather_calls() > 0);
        println!(
            "RECENT requested_rows={recent_rows} cache_aware_gathers={}",
            gpu.c4_recent_gather_calls()
        );
    }
    if capacity_test {
        let state = gpu
            .alloc_decode_state_host_c4(1_048_576, 512)
            .expect("1M direct host capacity");
        assert_eq!(
            state.host_cache_bytes,
            gpu.c4_host_bytes_for_capacity(1_048_576).unwrap()
        );
        println!(
            "CAPACITY_ONLY context=1048576 chunk=512 gpu_cache_bytes={:?} host_cache_bytes={:?} matrix_scratch_bytes={:?}",
            state.cache_bytes,
            state.host_cache_bytes,
            state.matrix_device_scratch_bytes()
        );
    }
    println!(
        "PASS direct host C4 creation/restoration identity; actual long prompts, serving, and concurrency remain separate"
    );
}
