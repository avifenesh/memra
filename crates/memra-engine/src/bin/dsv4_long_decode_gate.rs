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
        "usage: dsv4_long_decode_gate <model-dir> <real-source.txt> [prompt-tokens] [profile-prefill|host-c4] [chunk-rows]"
    );
    assert!(
        args.len() <= 6
            && args
                .get(4)
                .is_none_or(|a| a == "profile-prefill" || a == "host-c4"),
        "unknown gate option"
    );
    let host_c4 = args.get(4).is_some_and(|a| a == "host-c4");
    let chunk: usize = args
        .get(5)
        .map(|s| s.parse().expect("chunk rows"))
        .unwrap_or(32);
    assert!((1..=512).contains(&chunk), "chunk must be 1..512");
    assert!(
        host_c4 || chunk == 32,
        "legacy/profile protocol stays width 32"
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
    let mut state = if host_c4 {
        gpu.alloc_decode_state_host_c4(capacity, chunk)
    } else {
        gpu.alloc_decode_state_for_transient(capacity, chunk)
    }
    .expect("state");
    if host_c4 {
        assert_eq!(
            state.host_cache_bytes,
            gpu.c4_host_bytes_for_capacity(capacity)
                .expect("host forecast")
        );
        assert!(state.host_cache_bytes.iter().sum::<u64>() > 0);
    }
    println!(
        "ALLOCATION capacity={capacity} chunk={chunk} active_c4={host_c4} gpu_cache_bytes={:?} host_cache_bytes={:?}",
        state.cache_bytes, state.host_cache_bytes
    );
    let mut draft = gpu.dspark_alloc_state().expect("draft state");
    println!("CHECK real-source prefill tokens={count} chunk={chunk} active_c4={host_c4}");
    let prefill_start = Instant::now();
    let logits = if args.get(4).is_some_and(|arg| arg == "profile-prefill") {
        let begin = 16_384;
        let end = begin + 256;
        assert!(count > end, "profile needs a complete middle prefill span");
        gpu.dspark_prefill_prime_chunked(&prompt[..begin], &mut state, &mut draft, 32)
            .expect("pre-profile prefix");
        for stage in &gpu.stages {
            stage.gpu.stream().synchronize().expect("pre-profile drain");
        }
        gpu.stages[0]
            .gpu
            .ctx
            .bind_to_thread()
            .expect("profile context");
        cudarc::driver::safe::profiler_start().expect("profile start");
        gpu.dspark_continue_prefix_chunked(&prompt[begin..end], &mut state, &mut draft, 32)
            .expect("profile span");
        for stage in &gpu.stages {
            stage
                .gpu
                .stream()
                .synchronize()
                .expect("profile completion drain");
        }
        gpu.stages[0]
            .gpu
            .ctx
            .bind_to_thread()
            .expect("profile stop context");
        cudarc::driver::safe::profiler_stop().expect("profile stop");
        println!("PROFILE_COMPLETE prefill_positions={begin}..{end} both_stages_drained=true");
        gpu.dspark_continue_prefix_chunked(&prompt[end..], &mut state, &mut draft, 32)
            .expect("post-profile suffix")
    } else {
        gpu.dspark_prefill_prime_chunked(&prompt, &mut state, &mut draft, chunk)
            .expect("prefill")
    };
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
    let restore = || {
        if host_c4 {
            gpu.restore_decode_state_host_c4(&host, capacity, chunk)
        } else {
            gpu.restore_decode_state_for_transient(&host, capacity, chunk)
        }
    };
    let cfg = Dsv4SampleCfg {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        seed: 20260905,
    };
    println!("CHECK plain decode beyond 4096 compressed candidates");
    let plain_start = Instant::now();
    let plain = {
        let mut state = restore().expect("restore plain");
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
    let mut state = restore().expect("restore spec");
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
        "PASS real prompt={count} sampled_tokens=16 plain/spec identical rounds={} chunk={chunk} active_c4={host_c4}",
        run.rounds.len()
    );
}
