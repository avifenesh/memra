//! One-load exactness/engagement for final-row head and final-window draft prime.
use memra_engine::dsv4_gpu::{
    Dsv4Gpu, Dsv4PrefillDraft, Dsv4PrefillHead, Dsv4PrefillHeadStats, Dsv4SampleCfg, Dsv4Vt,
};
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::path::Path;
use std::time::Instant;

fn hash_floats(hash: &mut Sha256, values: &[f32], mask: bool) {
    hash.update((values.len() as u64).to_le_bytes());
    for value in values {
        assert!(value.is_finite() || (mask && *value == f32::NEG_INFINITY));
        hash.update(value.to_bits().to_le_bytes());
    }
}
fn hash_classes(hash: &mut Sha256, classes: Vec<(String, Vec<f32>)>) {
    assert!(!classes.is_empty());
    for (name, values) in classes {
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash_floats(
            hash,
            &values,
            name.ends_with(".cmp_pend_score") || name.ends_with(".idx_pend_score"),
        );
    }
}
fn expected(
    prefix: usize,
    suffix: usize,
    width: usize,
    head: Dsv4PrefillHead,
    draft: Dsv4PrefillDraft,
) -> Dsv4PrefillHeadStats {
    let mut result = Dsv4PrefillHeadStats::default();
    for len in [prefix - 1, suffix].into_iter().filter(|&len| len > 0) {
        match head {
            Dsv4PrefillHead::All => result.full_rows += len as u64,
            Dsv4PrefillHead::Last => {
                result.last_rows += 1;
                result.skipped_chunks += (len.div_ceil(width) - 1) as u64;
            }
        }
        result.draft_prime_rows += match draft {
            Dsv4PrefillDraft::All => len,
            Dsv4PrefillDraft::Tail => len.min(128),
        } as u64;
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn run(
    gpu: &Dsv4Gpu,
    tokens: &[u32],
    prefix: usize,
    suffix: usize,
    width: usize,
    head: Dsv4PrefillHead,
    draft_arm: Dsv4PrefillDraft,
) -> Vec<u8> {
    let count = prefix + suffix;
    let capacity = count + 64;
    // The prefill chunk can be one row, but the sampled tail still verifies a
    // full DSpark transaction. Preserve that space across snapshot/restore.
    let transient_rows = width.max(gpu.verify_tmax());
    let before = gpu.prefill_head_stats();
    let mut state = gpu
        .alloc_decode_state_for_transient(capacity, transient_rows)
        .expect("state");
    let mut draft = gpu.dspark_alloc_state().expect("draft");
    let timer = Instant::now();
    let mut logits = gpu
        .dspark_prefill_prime_chunked(&tokens[..prefix], &mut state, &mut draft, width)
        .expect("prime");
    let mut hash = Sha256::new();
    hash_floats(&mut hash, &logits, false);
    hash_classes(&mut hash, gpu.cache_classes(&state).expect("prime trunk"));
    hash_classes(
        &mut hash,
        gpu.dspark_ring_classes(&draft).expect("prime draft"),
    );
    if suffix > 0 {
        let host = gpu.snapshot_decode_state(&state).expect("snapshot trunk");
        let host_draft = gpu.snapshot_dspark_state(&draft).expect("snapshot draft");
        drop(state);
        drop(draft);
        state = gpu
            .restore_decode_state_for_transient(&host, capacity, transient_rows)
            .expect("restore trunk");
        draft = gpu
            .restore_dspark_state(&host_draft)
            .expect("restore draft");
        logits = gpu
            .dspark_continue_prefix_chunked(&tokens[prefix..count], &mut state, &mut draft, width)
            .expect("restored suffix");
    }
    let wall = timer.elapsed().as_secs_f64();
    let after = gpu.prefill_head_stats();
    let observed = Dsv4PrefillHeadStats {
        full_rows: after.full_rows - before.full_rows,
        last_rows: after.last_rows - before.last_rows,
        skipped_chunks: after.skipped_chunks - before.skipped_chunks,
        draft_prime_rows: after.draft_prime_rows - before.draft_prime_rows,
    };
    assert_eq!(
        observed,
        expected(prefix, suffix, width, head, draft_arm),
        "engagement"
    );
    println!(
        "WORK prefix={prefix} suffix={suffix} width={width} head={head:?} draft={draft_arm:?} gate_wall_including_checks_s={wall:.6} counters={observed:?}"
    );
    hash_floats(&mut hash, &logits, false);
    hash_classes(&mut hash, gpu.cache_classes(&state).expect("trunk"));
    hash_classes(&mut hash, gpu.dspark_ring_classes(&draft).expect("draft"));
    if count > 1 {
        let cfg = Dsv4SampleCfg {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            seed: 20260905,
        };
        let mut verify = gpu.alloc_verify_state_for(capacity).expect("verify");
        let sample = gpu
            .spec_sampled_batched_pen_restored(
                &tokens[..count],
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
        assert_eq!(sample.tokens.len(), 16);
        assert!(!sample.rounds.is_empty());
        for token in sample.tokens {
            hash.update(token.to_le_bytes());
        }
        for round in sample.rounds {
            for value in [
                round.start_pos,
                round.accepts,
                round.verified,
                round.t_batch,
                round.emitted,
            ] {
                hash.update((value as u64).to_le_bytes());
            }
            hash_floats(&mut hash, &round.confidence, false);
        }
        hash_classes(&mut hash, gpu.cache_classes(&state).expect("final trunk"));
        hash_classes(
            &mut hash,
            gpu.dspark_ring_classes(&draft).expect("final draft"),
        );
    }
    hash.finalize().to_vec()
}

fn public_verify_contract(gpu: &Dsv4Gpu, tokens: &[u32]) {
    let mut state = gpu.alloc_decode_state_for_transient(96, 8).expect("state");
    gpu.prefill_with_cache_chunked(&tokens[..32], &mut state, 8)
        .expect("prime");
    // Deliberately use a prefill workspace: phase metadata must not override the
    // public verifier's explicit full-row/argmax request.
    let mut verify = gpu.alloc_prefill_state_for(96, 8).expect("workspace");
    let (rows, argmax) = gpu
        .verify_batch_dev(&tokens[32..35], &mut state, &mut verify, None, true)
        .expect("full rows");
    assert_eq!(
        rows.expect("full logits").len(),
        3 * gpu
            .stages
            .last()
            .expect("stage")
            .head
            .as_ref()
            .expect("head")
            .len()
            / 2
            / 4096
    );
    assert_eq!(argmax.len(), 3);
    gpu.commit_verify_dev(&mut state, &mut verify, 3)
        .expect("commit full");
    let (rows, argmax) = gpu
        .verify_batch_dev(&tokens[35..38], &mut state, &mut verify, None, false)
        .expect("argmax rows");
    assert!(rows.is_none());
    assert_eq!(argmax.len(), 3);
    gpu.commit_verify_dev(&mut state, &mut verify, 3)
        .expect("commit argmax");
    println!("PASS public full-row and argmax contract under Last policy");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(
        args.len(),
        3,
        "usage: dsv4_prefill_work_gate <model-dir> <real-source.txt>"
    );
    let dir = Path::new(&args[1]);
    let source = std::fs::read_to_string(&args[2]).expect("source");
    let tokenizer = Tokenizer::from_hf_dir(dir).expect("tokenizer");
    let mut tokens = tokenizer.encode(
        &format!("Review this inference engine source:\n{source}"),
        true,
    );
    assert!(tokens.len() >= 1600, "real source required; no padding");
    tokens.truncate(1600);
    println!("SOURCE sha256={:x}", Sha256::digest(source.as_bytes()));
    let mut gpu = Dsv4Gpu::load(dir, &[0, 1], ActQuantVariant::RefFp8Round, 4096).expect("load");
    gpu.set_prefill_grouped_for_gate(false)
        .expect("qualified reference expert math only");
    for (prefix, suffix, width) in [
        (1, 0, 32),
        (32, 0, 1),
        (160, 0, 32),
        (1025, 0, 64),
        (1025, 0, 512),
        (160, 33, 32),
        (160, 128, 32),
        (160, 129, 32),
        (1025, 257, 512),
    ] {
        gpu.set_prefill_head_for_gate(Dsv4PrefillHead::All)
            .expect("all heads");
        gpu.set_prefill_draft_for_gate(Dsv4PrefillDraft::All)
            .expect("all draft");
        let reference = run(
            &gpu,
            &tokens,
            prefix,
            suffix,
            width,
            Dsv4PrefillHead::All,
            Dsv4PrefillDraft::All,
        );
        for (head, draft) in [
            (Dsv4PrefillHead::Last, Dsv4PrefillDraft::All),
            (Dsv4PrefillHead::All, Dsv4PrefillDraft::Tail),
            (Dsv4PrefillHead::Last, Dsv4PrefillDraft::Tail),
        ] {
            gpu.set_prefill_head_for_gate(head).expect("head arm");
            gpu.set_prefill_draft_for_gate(draft).expect("draft arm");
            assert_eq!(
                reference,
                run(&gpu, &tokens, prefix, suffix, width, head, draft),
                "prefix={prefix} suffix={suffix} width={width} head={head:?} draft={draft:?}"
            );
            println!(
                "EXACT prefix={prefix} suffix={suffix} width={width} head={head:?} draft={draft:?}"
            );
        }
    }
    public_verify_contract(&gpu, &tokens);
    println!(
        "PASS prefill work elision: logits/cache/DSpark/sampled/engagement; performance promotion not claimed"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn engagement_distinguishes_full_program_from_last_only() {
        let all = expected(1025, 129, 512, Dsv4PrefillHead::All, Dsv4PrefillDraft::All);
        let last = expected(
            1025,
            129,
            512,
            Dsv4PrefillHead::Last,
            Dsv4PrefillDraft::Tail,
        );
        assert_eq!((all.full_rows, all.draft_prime_rows), (1153, 1153));
        assert_eq!(
            (last.last_rows, last.skipped_chunks, last.draft_prime_rows),
            (2, 1, 256)
        );
        assert_ne!(all, last);
    }
}
