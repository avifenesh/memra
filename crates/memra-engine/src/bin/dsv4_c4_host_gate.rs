//! Active C4 residency: compare to all-device state under identical numeric policy.
use memra_engine::dsv4_gpu::{Dsv4Gpu, Dsv4SampleCfg, Dsv4Vt, dsv4_sample_row};
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::path::Path;

fn digest(classes: Vec<(String, Vec<f32>)>) -> Vec<u8> {
    let mut hash = Sha256::new();
    for (name, values) in classes {
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update((values.len() as u64).to_le_bytes());
        for v in values {
            hash.update(v.to_bits().to_le_bytes());
        }
    }
    hash.finalize().to_vec()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(
        args.len(),
        3,
        "usage: dsv4_c4_host_gate <model-dir> <source.txt>"
    );
    let dir = Path::new(&args[1]);
    let source = std::fs::read_to_string(&args[2]).expect("source");
    let tokenizer = Tokenizer::from_hf_dir(dir).expect("tokenizer");
    let tokens = tokenizer.encode(&format!("Review this inference code:\n{source}"), true);
    assert!(tokens.len() > 5000);
    let gpu = Dsv4Gpu::load(dir, &[0, 1], ActQuantVariant::RefFp8Round, 8192).expect("load");
    let cfg = Dsv4SampleCfg {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        seed: 20260905,
    };
    for count in [1, 160, 4097] {
        let capacity = count + 512;
        let mut state = gpu
            .alloc_decode_state_for_transient(capacity, 32)
            .expect("state");
        let mut draft = gpu.dspark_alloc_state().expect("draft");
        let prime = gpu
            .dspark_prefill_prime_chunked(&tokens[..count], &mut state, &mut draft, 32)
            .expect("prime");
        let snapshot = gpu.snapshot_decode_state(&state).expect("snapshot");
        let snapshot_draft = gpu.snapshot_dspark_state(&draft).expect("draft snapshot");
        let original_digest = digest(gpu.cache_classes(&state).expect("original classes"));
        drop(state);
        drop(draft);
        for suffix in [0, 33, 129] {
            let mut reference = None;
            for offload in [false, true, false] {
                let mut state = gpu
                    .restore_decode_state_for_transient(&snapshot, capacity, 32)
                    .expect("restore");
                let mut draft = gpu
                    .restore_dspark_state(&snapshot_draft)
                    .expect("restore draft");
                let mut verify = gpu.alloc_verify_state_for(capacity).expect("verify");
                if offload {
                    let before = state.cache_bytes.clone();
                    let freed = gpu
                        .offload_c4_decode_state(&mut state, &verify)
                        .expect("offload");
                    assert!(freed.iter().sum::<u64>() > 0, "C4 must leave VRAM");
                    for stage in 0..freed.len() {
                        assert_eq!(before[stage] - state.cache_bytes[stage], freed[stage]);
                        assert_eq!(freed[stage], state.host_cache_bytes[stage]);
                    }
                    assert_eq!(
                        original_digest,
                        digest(gpu.cache_classes(&state).expect("offloaded classes"))
                    );
                    assert!(
                        gpu.offload_c4_decode_state(&mut state, &verify)
                            .expect("idempotent")
                            .iter()
                            .all(|b| *b == 0)
                    );
                    println!(
                        "RESIDENCY count={count} released_device_bytes={freed:?} active_host_bytes={:?}",
                        state.host_cache_bytes
                    );
                }
                let mut row = prime.clone();
                if suffix > 0 {
                    row = gpu
                        .dspark_continue_prefix_chunked(
                            &tokens[count..count + suffix],
                            &mut state,
                            &mut draft,
                            32,
                        )
                        .expect("offloaded suffix");
                }
                // Snapshot from active host residency must restore the canonical
                // all-device representation with identical live values.
                let parked = gpu.snapshot_decode_state(&state).expect("park active C4");
                let restored = gpu
                    .restore_decode_state_for_transient(&parked, capacity, 32)
                    .expect("restore active snapshot");
                assert_eq!(
                    digest(gpu.cache_classes(&state).expect("park source")),
                    digest(gpu.cache_classes(&restored).expect("park destination"))
                );
                drop(restored);
                drop(parked);
                let mut hash = Sha256::new();
                for v in &row {
                    hash.update(v.to_bits().to_le_bytes());
                }
                // Plain decode exercises compressed-row emissions and mapped reads
                // before a speculative pass exercises rollback and re-emission.
                let mut prompt = tokens[..count + suffix].to_vec();
                for _ in 0..9 {
                    let token = dsv4_sample_row(&row, state.pos, &cfg).expect("sample");
                    prompt.push(token);
                    let pos = state.pos;
                    row = gpu
                        .decode_step_tap(token, &mut state, &mut draft, 0)
                        .expect("plain host decode");
                    gpu.dspark_write_rings(&mut draft, 0, pos)
                        .expect("plain draft rings");
                    hash.update(token.to_le_bytes());
                    for v in &row {
                        hash.update(v.to_bits().to_le_bytes());
                    }
                }
                let run = gpu
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
                    .expect("host spec");
                assert_eq!(run.tokens.len(), 32);
                assert!(!run.rounds.is_empty());
                for token in run.tokens {
                    hash.update(token.to_le_bytes());
                }
                for round in &run.rounds {
                    for value in [
                        round.start_pos,
                        round.accepts,
                        round.verified,
                        round.t_batch,
                        round.emitted,
                    ] {
                        hash.update((value as u64).to_le_bytes());
                    }
                }
                hash.update(digest(gpu.cache_classes(&state).expect("final trunk")));
                hash.update(digest(
                    gpu.dspark_ring_classes(&draft).expect("final draft"),
                ));
                let result = hash.finalize().to_vec();
                if let Some(reference) = &reference {
                    assert_eq!(
                        reference, &result,
                        "count={count} suffix={suffix} host={offload}"
                    );
                } else {
                    reference = Some(result);
                }
                println!(
                    "EXACT active C4 count={count} suffix={suffix} host={offload} sampled=41 rounds={}",
                    run.rounds.len()
                );
            }
        }
    }
    println!(
        "PASS active C4 device/host/device logits, sampled tokens, rollback, suffix and snapshot identity; serving and performance remain unqualified"
    );
}
