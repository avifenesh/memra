//! One-load, finite and bit-exact block/warp FP4 rewrite gate for the 0731 mint.
//! No timing verdict: serving A/B, long-context and concurrency remain separate.
use memra_engine::dsv4_gpu::{
    Dsv4Fp4Reduce, Dsv4Gpu, Dsv4IndexerScore, Dsv4SampleCfg, Dsv4Vt, SpecRunGpu, dsv4_sample_row,
};
use memra_gguf::dsv4_forward::FixtureSpec;
use sha2::{Digest, Sha256};
use std::path::Path;

fn hash_floats(hash: &mut Sha256, values: &[f32]) {
    hash.update((values.len() as u64).to_le_bytes());
    for value in values {
        assert!(value.is_finite(), "non-finite gate operand");
        hash.update(value.to_bits().to_le_bytes());
    }
}

fn hash_classes(hash: &mut Sha256, classes: Vec<(String, Vec<f32>)>) {
    assert!(!classes.is_empty(), "no live cache classes");
    for (name, values) in classes {
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        if name.ends_with(".cmp_pend_score") || name.ends_with(".idx_pend_score") {
            // alloc_decode_state_for_transient initializes these score planes to
            // -inf for not-yet-populated compressor positions. It is a semantic
            // mask, not numeric poison, and its exact bits must still compare.
            hash.update((values.len() as u64).to_le_bytes());
            for (index, value) in values.iter().enumerate() {
                assert!(
                    value.is_finite() || *value == f32::NEG_INFINITY,
                    "invalid pending score: {name}[{index}] bits={:08x}",
                    value.to_bits()
                );
                hash.update(value.to_bits().to_le_bytes());
            }
        } else {
            for (index, value) in values.iter().enumerate() {
                assert!(
                    value.is_finite(),
                    "non-finite cache: {name}[{index}] bits={:08x}",
                    value.to_bits()
                );
            }
            hash_floats(hash, &values);
        }
    }
}

fn plain(gpu: &Dsv4Gpu, prompt: &[u32], width: usize, sample: &Dsv4SampleCfg) -> Vec<u8> {
    let mut state = gpu
        .alloc_decode_state_for_transient(prompt.len() + 32, width)
        .expect("plain cache allocation");
    let mut logits = gpu
        .prefill_with_cache_chunked(prompt, &mut state, width)
        .expect("chunked prefill");
    let mut hash = Sha256::new();
    hash_classes(
        &mut hash,
        gpu.cache_classes(&state).expect("prefill classes"),
    );
    for step in 0..16 {
        assert!(!logits.is_empty(), "empty logits");
        hash_floats(&mut hash, &logits);
        let token = dsv4_sample_row(&logits, prompt.len() + step, sample).expect("sample");
        hash.update(token.to_le_bytes());
        logits = gpu.decode_step(token, &mut state).expect("plain decode");
    }
    hash_floats(&mut hash, &logits);
    hash_classes(
        &mut hash,
        gpu.cache_classes(&state).expect("decode classes"),
    );
    hash.finalize().to_vec()
}

fn hash_rounds(hash: &mut Sha256, run: &SpecRunGpu) {
    assert_eq!(run.tokens.len(), 16, "sampled output count");
    assert!(!run.rounds.is_empty(), "DSpark never engaged");
    assert!(run.rounds.iter().any(|r| !r.drafts.is_empty()), "no drafts");
    for token in &run.tokens {
        hash.update(token.to_le_bytes());
    }
    for round in &run.rounds {
        for n in [
            round.start_pos,
            round.accepts,
            round.verified,
            round.t_batch,
            round.t_cap,
            round.emitted,
        ] {
            hash.update((n as u64).to_le_bytes());
        }
        hash.update((round.drafts.len() as u64).to_le_bytes());
        for token in &round.drafts {
            hash.update(token.to_le_bytes());
        }
        hash_floats(hash, &round.confidence);
        // round_us intentionally excluded: this is an exactness instrument.
    }
}

fn speculative(gpu: &Dsv4Gpu, prompt: &[u32], sample: &Dsv4SampleCfg) -> Vec<u8> {
    let mut state = gpu.alloc_decode_state().expect("spec cache");
    let mut dstate = gpu.dspark_alloc_state().expect("draft state");
    let mut vstate = gpu.alloc_verify_state().expect("verify state");
    let run = gpu
        .spec_sampled_batched_policy(
            prompt,
            16,
            &mut state,
            &mut dstate,
            &mut vstate,
            usize::MAX,
            Dsv4Vt::Off,
            sample,
        )
        .expect("sampled spec run");
    let mut hash = Sha256::new();
    hash_rounds(&mut hash, &run);
    hash_classes(
        &mut hash,
        gpu.cache_classes(&state).expect("spec trunk classes"),
    );
    hash_classes(
        &mut hash,
        gpu.dspark_ring_classes(&dstate)
            .expect("draft ring classes"),
    );
    hash.finalize().to_vec()
}

fn set_arm(gpu: &mut Dsv4Gpu, indexer: bool, candidate: bool) {
    if indexer {
        gpu.set_indexer_score_for_gate(if candidate {
            Dsv4IndexerScore::Tiled
        } else {
            Dsv4IndexerScore::Scalar
        })
        .expect("indexer gate arm");
    } else {
        gpu.set_fp4_reduce_for_gate(if candidate {
            Dsv4Fp4Reduce::Warp
        } else {
            Dsv4Fp4Reduce::Block
        })
        .expect("FP4 gate arm");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(
        args.len() == 3 || (args.len() == 4 && args[3] == "indexer"),
        "usage: dsv4_fp4_reduce_gate <model-dir> <0731-fixtures.json> [indexer]"
    );
    let indexer = args.len() == 4;
    let axis = if indexer {
        "indexer scalar/tiled"
    } else {
        "FP4 block/warp"
    };
    println!("AXIS {axis}");
    for (name, required) in [
        ("MEMRA_DSV4_DECODE_PATH", "device"),
        ("MEMRA_DSV4_EXPERT_ARM", "native"),
        ("MEMRA_DSV4_DRAFTER", "dspark"),
    ] {
        assert_eq!(
            std::env::var(name).as_deref(),
            Ok(required),
            "requires {name}={required}"
        );
    }
    let fixture = FixtureSpec::load(Path::new(&args[2]));
    assert_eq!(
        fixture.variant_tag, "ref",
        "0731 reference activation contract required"
    );
    let long = fixture
        .tokens_160
        .as_ref()
        .expect("160-token real fixture required");
    assert!(
        long.len() >= 130,
        "must exercise multiple full width-64 chunks"
    );
    assert!(!fixture.tokens_32.is_empty());
    let mut gpu = Dsv4Gpu::load(
        Path::new(&args[1]),
        &[0, 1],
        fixture.variant,
        long.len() + 64,
    )
    .expect("load pinned 0731 model");
    assert!(gpu.dspark.is_some(), "bundled DSpark required");
    // Exact artifact generation_config.json: temperature=1.0, top_p=1.0, no top-k.
    let sample = Dsv4SampleCfg {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        seed: 20260905,
    };
    let original = gpu
        .set_fp4_reduce_for_gate(Dsv4Fp4Reduce::Block)
        .expect("arm block");
    let original_fused = gpu.dspark_fused_moe;
    let original_indexer = if indexer {
        Some(
            gpu.set_indexer_score_for_gate(Dsv4IndexerScore::Scalar)
                .expect("arm scalar"),
        )
    } else {
        None
    };
    for prompt in [&fixture.tokens_32, long] {
        for width in [1, 32, 64] {
            println!(
                "CHECK plain prompt={} width={width} {axis} reference/candidate/reference",
                prompt.len()
            );
            set_arm(&mut gpu, indexer, false);
            let reference = plain(&gpu, prompt, width, &sample);
            set_arm(&mut gpu, indexer, true);
            assert_eq!(
                reference,
                plain(&gpu, prompt, width, &sample),
                "plain width={width}"
            );
            // Prove same-process arm reversal as well as A->B.
            set_arm(&mut gpu, indexer, false);
            assert_eq!(
                reference,
                plain(&gpu, prompt, width, &sample),
                "reference restoration"
            );
            println!(
                "EXACT plain prompt={} width={width} {axis} reference/candidate/reference",
                prompt.len()
            );
        }
        for fused in [false, true] {
            println!("CHECK DSpark prompt={} fused={fused} {axis}", prompt.len());
            gpu.dspark_fused_moe = fused;
            set_arm(&mut gpu, indexer, false);
            let reference = speculative(&gpu, prompt, &sample);
            set_arm(&mut gpu, indexer, true);
            assert_eq!(
                reference,
                speculative(&gpu, prompt, &sample),
                "spec fused={fused}"
            );
            println!(
                "EXACT DSpark prompt={} fused={fused} tokens/rounds/confidence/cache",
                prompt.len()
            );
        }
    }
    gpu.dspark_fused_moe = original_fused;
    if let Some(original) = original_indexer {
        gpu.set_indexer_score_for_gate(original)
            .expect("restore indexer arm");
    }
    gpu.set_fp4_reduce_for_gate(original)
        .expect("restore original arm");
    println!("PASS {axis} rewrite one-load exactness; serving/performance not claimed");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_digest_preserves_bits() {
        let digest = |v| {
            let mut h = Sha256::new();
            hash_floats(&mut h, &[v]);
            h.finalize()
        };
        assert_ne!(digest(1.0), digest(f32::from_bits(1.0f32.to_bits() + 1)));
        assert_ne!(digest(0.0), digest(-0.0));
    }

    #[test]
    #[should_panic(expected = "non-finite gate operand")]
    fn poisoned_equal_bits_refuse() {
        hash_floats(&mut Sha256::new(), &[f32::NAN]);
    }

    #[test]
    #[should_panic(expected = "no live cache classes")]
    fn empty_cache_census_refuses() {
        hash_classes(&mut Sha256::new(), vec![]);
    }

    #[test]
    fn pending_score_negative_infinity_is_a_hashed_mask() {
        let digest = |value| {
            let mut h = Sha256::new();
            hash_classes(&mut h, vec![("l0.cmp_pend_score".into(), vec![value])]);
            h.finalize()
        };
        assert_ne!(digest(f32::NEG_INFINITY), digest(0.0));
    }

    #[test]
    #[should_panic(expected = "non-finite cache")]
    fn negative_infinity_in_kv_refuses() {
        hash_classes(
            &mut Sha256::new(),
            vec![("l0.ring".into(), vec![f32::NEG_INFINITY])],
        );
    }

    #[test]
    #[should_panic(expected = "invalid pending score")]
    fn nan_in_pending_scores_refuses() {
        hash_classes(
            &mut Sha256::new(),
            vec![("l0.cmp_pend_score".into(), vec![f32::NAN])],
        );
    }

    #[test]
    #[should_panic(expected = "invalid pending score")]
    fn positive_infinity_in_pending_scores_refuses() {
        hash_classes(
            &mut Sha256::new(),
            vec![("l0.idx_pend_score".into(), vec![f32::INFINITY])],
        );
    }
}
