//! Real-source prompt across the former 16K decode-selector boundary.
//! One prefill, exact host snapshot/restore, then sampled plain and DSpark walks.
use memra_engine::dsv4_gpu::{Dsv4Gpu, Dsv4SampleCfg, Dsv4Vt, dsv4_sample_row};
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::path::Path;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(
        args.len() >= 3,
        "usage: dsv4_long_decode_gate <model-dir> <real-source.txt> [prompt-tokens]"
    );
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
    let count: usize = args
        .get(3)
        .map(|s| s.parse().expect("prompt count"))
        .unwrap_or(16_416);
    assert!(
        count > 16_384 && count <= 1_048_448,
        "prompt must cross the old boundary and leave output room"
    );
    let dir = Path::new(&args[1]);
    let text = std::fs::read_to_string(&args[2]).expect("read real source");
    let tokenizer = Tokenizer::from_hf_dir(dir).expect("pinned tokenizer");
    let mut prompt = tokenizer.encode(
        &format!("Review this inference engine source:\n\n{text}"),
        true,
    );
    assert!(
        prompt.len() >= count,
        "source is too short; do not repeat/pad it"
    );
    prompt.truncate(count);
    let mut hash = Sha256::new();
    for token in &prompt {
        hash.update(token.to_le_bytes());
    }
    println!(
        "INPUT tokens={count} sha256={:x} source_sha256={:x}",
        hash.finalize(),
        Sha256::digest(text.as_bytes())
    );

    let capacity = count + 32;
    let gpu = Dsv4Gpu::load(dir, &[0, 1], ActQuantVariant::RefFp8Round, capacity)
        .expect("load 0731 model");
    let mut state = gpu
        .alloc_decode_state_for_transient(capacity, 32)
        .expect("state");
    let mut draft = gpu.dspark_alloc_state().expect("draft state");
    println!("CHECK real-source prefill tokens={count} chunk=32");
    let prefill_start = Instant::now();
    let logits = gpu
        .dspark_prefill_prime_chunked(&prompt, &mut state, &mut draft, 32)
        .expect("prefill");
    let prefill_s = prefill_start.elapsed().as_secs_f64();
    println!(
        "PREFILL tokens={count} seconds={prefill_s:.6} tokens_per_second={:.3}",
        count as f64 / prefill_s
    );
    assert!(
        logits.iter().all(|x| x.is_finite()),
        "non-finite prefill logits"
    );
    let host = gpu.snapshot_decode_state(&state).expect("snapshot trunk");
    let host_draft = gpu.snapshot_dspark_state(&draft).expect("snapshot draft");
    drop(state);
    drop(draft);
    let cfg = Dsv4SampleCfg {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        seed: 20260905,
    };
    println!("CHECK plain decode beyond 4096 compressed candidates");
    let plain_start = Instant::now();
    let plain = {
        let mut state = gpu
            .restore_decode_state_for_transient(&host, capacity, 32)
            .expect("restore plain");
        let mut row = logits.clone();
        let mut tokens = Vec::new();
        for i in 0..16 {
            assert!(row.iter().all(|x| x.is_finite()), "non-finite plain logits");
            let token = dsv4_sample_row(&row, count + i, &cfg).expect("plain sample");
            tokens.push(token);
            if i < 15 {
                row = gpu
                    .decode_step(token, &mut state)
                    .expect("long plain decode");
            }
        }
        tokens
    };
    println!(
        "PLAIN_WITH_RESTORE seconds={:.6}",
        plain_start.elapsed().as_secs_f64()
    );
    println!("CHECK narrow DSpark verify beyond 4096 compressed candidates");
    let spec_start = Instant::now();
    let mut state = gpu
        .restore_decode_state_for_transient(&host, capacity, 32)
        .expect("restore spec");
    let mut draft = gpu
        .restore_dspark_state(&host_draft)
        .expect("restore draft");
    let mut verify = gpu.alloc_verify_state().expect("verify workspace");
    let run = gpu
        .spec_sampled_batched_pen_restored(
            &prompt,
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
        .expect("long sampled spec");
    println!(
        "SPEC_WITH_RESTORE seconds={:.6}",
        spec_start.elapsed().as_secs_f64()
    );
    assert_eq!(plain, run.tokens, "sampled plain/spec output mismatch");
    assert!(!run.rounds.is_empty(), "DSpark did not engage");
    println!(
        "PASS real prompt={count} sampled_tokens=16 plain/spec identical rounds={}",
        run.rounds.len()
    );
}
