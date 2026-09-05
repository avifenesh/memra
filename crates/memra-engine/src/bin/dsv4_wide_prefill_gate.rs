//! Exact width-32/128/512 transaction comparison on a real 1025-token prompt.
//! Includes every live cache class, DSpark rings and sampled speculative output.
use memra_engine::dsv4_gpu::{Dsv4Gpu, Dsv4SampleCfg, Dsv4Vt};
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::path::Path;
use std::time::Instant;

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
fn run(gpu: &Dsv4Gpu, prompt: &[u32], width: usize) -> Vec<u8> {
    let mut state = gpu
        .alloc_decode_state_for_transient(prompt.len() + 32, width)
        .expect("cache");
    let mut draft = gpu.dspark_alloc_state().expect("draft");
    let timer = Instant::now();
    let logits = gpu
        .dspark_prefill_prime_chunked(prompt, &mut state, &mut draft, width)
        .expect("wide prime");
    println!(
        "PREFILL width={width} tokens={} seconds={:.6}",
        prompt.len(),
        timer.elapsed().as_secs_f64()
    );
    let mut hash = Sha256::new();
    floats(&mut hash, &logits, false);
    classes(&mut hash, gpu.cache_classes(&state).expect("trunk classes"));
    classes(
        &mut hash,
        gpu.dspark_ring_classes(&draft).expect("draft classes"),
    );
    let cfg = Dsv4SampleCfg {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        seed: 20260905,
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
    hash.finalize().to_vec()
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(
        args.len(),
        3,
        "usage: dsv4_wide_prefill_gate <model-dir> <real-source.txt>"
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
    for grouped in [false, true] {
        gpu.set_prefill_grouped_for_gate(grouped).expect("arm");
        println!("CHECK grouped={grouped} widths=32/128/512");
        let reference = run(&gpu, &prompt, 32);
        for width in [128, 512] {
            assert_eq!(
                reference,
                run(&gpu, &prompt, width),
                "width={width} grouped={grouped}"
            );
            println!("EXACT grouped={grouped} width={width} logits/live-cache/DSpark/sampled");
        }
    }
    println!("PASS wide transaction identity; 1M and serving qualification remain separate");
}
