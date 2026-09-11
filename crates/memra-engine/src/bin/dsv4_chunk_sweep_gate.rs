//! One-load matrix+EP chunk discovery: exact live state and sampled continuation.
//! Timed section is engine prefill only, not HTTP TTFT or output throughput.
use memra_engine::dsv4_gpu::{Dsv4Gpu, Dsv4SampleCfg, Dsv4Vt};
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::{
    path::Path,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("UTC clock")
        .as_millis()
}

fn floats(hash: &mut Sha256, values: &[f32], mask: bool) {
    hash.update((values.len() as u64).to_le_bytes());
    for value in values {
        assert!(value.is_finite() || (mask && *value == f32::NEG_INFINITY));
        hash.update(value.to_le_bytes());
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

fn run(gpu: &Dsv4Gpu, prompt: &[u32], width: usize, ordinal: usize) -> Vec<u8> {
    let capacity = prompt.len() + 32;
    let mut state = gpu
        .alloc_decode_state_for_transient(capacity, width.max(gpu.verify_tmax()))
        .expect("state");
    let mut draft = gpu.dspark_alloc_state().expect("draft");
    for stage in &gpu.stages {
        stage.gpu.stream().synchronize().expect("pre-timing drain");
    }
    let ep_before = gpu.ep_calls();
    let routes_before = gpu.grouped_device_route_calls();
    let start_unix_ms = unix_ms();
    println!(
        "START ordinal={ordinal} width={width} prompt={} matrix=true EP=pair start_unix_ms={start_unix_ms}",
        prompt.len()
    );
    let timer = Instant::now();
    let logits = gpu
        .dspark_prefill_prime_chunked(prompt, &mut state, &mut draft, width)
        .expect("prefill");
    for stage in &gpu.stages {
        stage
            .gpu
            .stream()
            .synchronize()
            .expect("prefill completion drain");
    }
    let seconds = timer.elapsed().as_secs_f64();
    let end_unix_ms = unix_ms();
    let ep_calls = gpu.ep_calls() - ep_before;
    let route_calls = gpu.grouped_device_route_calls() - routes_before;
    assert!(ep_calls > 0 && route_calls > 0, "matrix/EP did not engage");
    let mut hash = Sha256::new();
    floats(&mut hash, &logits, false);
    classes(&mut hash, gpu.cache_classes(&state).expect("prefill cache"));
    classes(
        &mut hash,
        gpu.dspark_ring_classes(&draft).expect("prefill draft"),
    );
    let mut verify = gpu.alloc_verify_state_for(capacity).expect("verify");
    let cfg = Dsv4SampleCfg {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        seed: 20260906,
    };
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
        .expect("sampled continuation");
    assert_eq!(sampled.tokens.len(), 16);
    assert!(!sampled.rounds.is_empty(), "DSpark not engaged");
    for token in &sampled.tokens {
        hash.update(token.to_le_bytes());
    }
    for round in &sampled.rounds {
        for n in [
            round.start_pos,
            round.accepts,
            round.verified,
            round.t_batch,
            round.emitted,
        ] {
            hash.update((n as u64).to_le_bytes());
        }
        floats(&mut hash, &round.confidence, false);
    }
    classes(&mut hash, gpu.cache_classes(&state).expect("final cache"));
    classes(
        &mut hash,
        gpu.dspark_ring_classes(&draft).expect("final draft"),
    );
    let signature = hash.finalize();
    println!(
        "MEASURE ordinal={ordinal} width={width} prompt={} prefill_seconds={seconds:.6} input_tokens_per_second={:.3} ep_calls={ep_calls} device_route_calls={route_calls} start_unix_ms={start_unix_ms} end_unix_ms={end_unix_ms} sampled_tokens={:?} rounds={} signature={signature:x}",
        prompt.len(),
        prompt.len() as f64 / seconds,
        sampled.tokens,
        sampled.rounds.len()
    );
    signature.to_vec()
}

fn parse_count(raw: Option<&str>) -> Result<usize, &'static str> {
    let count = raw
        .unwrap_or("9900")
        .parse::<usize>()
        .map_err(|_| "invalid prompt count")?;
    if !(1025..=1_048_448).contains(&count) {
        return Err("prompt count outside supported gate range");
    }
    Ok(count)
}

fn schedule(compare: Option<&str>) -> Result<Vec<usize>, &'static str> {
    let Some(raw) = compare else {
        return Ok(vec![32, 64, 128, 256, 512, 32]);
    };
    let width = raw
        .parse::<usize>()
        .map_err(|_| "invalid comparison width")?;
    if ![64, 128, 256, 512].contains(&width) {
        return Err("comparison width must be 64/128/256/512");
    }
    Ok([32, width, width, 32].repeat(3))
}

fn main() {
    // Freeze this historical instrument independently of the newer defaults.
    // This is process startup, before any model or worker threads exist.
    unsafe {
        std::env::set_var("MEMRA_DSV4_HC_DOT_SPLIT", "0");
        std::env::set_var("MEMRA_DSV4_DENSE_FAST", "0");
        // Gate-only AR phase instrument: pinned off here so no other bin can inherit
        // an exported instrument or null collective from the environment.
        std::env::set_var("MEMRA_DSV4_AR_PHASE", "0");
    }

    let args: Vec<String> = std::env::args().collect();
    assert!(
        matches!(args.len(), 3 | 4 | 6),
        "usage: dsv4_chunk_sweep_gate <model-dir> <real-source.txt> [prompt-tokens] [compare WIDTH]"
    );
    let compare = if args.len() == 6 {
        assert_eq!(args[4], "compare", "unknown comparison option");
        Some(args[5].as_str())
    } else {
        None
    };
    let widths = schedule(compare).expect("width schedule");
    println!(
        "PROTOCOL mode={} widths={widths:?} fresh_state_per_case=true timed=engine_prefill_only",
        if compare.is_some() {
            "ABBAx3_N6_each"
        } else {
            "discovery"
        }
    );
    for (name, required) in [
        ("MEMRA_DSV4_DECODE_PATH", "device"),
        ("MEMRA_DSV4_EXPERT_ARM", "native"),
        ("MEMRA_DSV4_DRAFTER", "dspark"),
        ("MEMRA_DSV4_DENSE_ARM", "fp8"),
        ("MEMRA_DSV4_EP", "pair"),
        ("MEMRA_DSV4_GROUPED_ROUTE", "device"),
        ("MEMRA_DSV4_PREFILL_MOE", "reference"),
    ] {
        assert_eq!(
            std::env::var(name).as_deref(),
            Ok(required),
            "requires {name}={required}"
        );
    }
    let count = parse_count(args.get(3).map(String::as_str)).expect("prompt count");
    let dir = Path::new(&args[1]);
    let source = std::fs::read_to_string(&args[2]).expect("real source");
    let tokenizer = Tokenizer::from_hf_dir(dir).expect("tokenizer");
    let mut prompt = tokenizer.encode(
        &format!("Review this inference engine source:\n\n{source}"),
        true,
    );
    assert!(prompt.len() >= count, "do not pad or repeat source");
    prompt.truncate(count);
    let mut token_hash = Sha256::new();
    for token in &prompt {
        token_hash.update(token.to_le_bytes());
    }
    println!(
        "INPUT count={count} token_sha256={:x} source_sha256={:x}",
        token_hash.finalize(),
        Sha256::digest(source.as_bytes())
    );
    // memra #458: this is a bench process, so it may run the matrix expert program
    // with the default-ON split-K arm; a serving process cannot arm it and refuses
    // that combination at load instead of failing every request.
    let gpu = Dsv4Gpu::load(dir, &[0, 1], ActQuantVariant::RefFp8Round, count + 32)
        .expect("matrix/EP load");
    assert!(gpu.matrix_moe_enabled());
    let mut baseline = None;
    for (ordinal, width) in widths.into_iter().enumerate() {
        let signature = run(&gpu, &prompt, width, ordinal);
        if let Some(expected) = &baseline {
            assert_eq!(
                expected, &signature,
                "chunk width changed the request program at width={width}"
            );
        } else {
            baseline = Some(signature);
        }
        for stage in &gpu.stages {
            stage
                .gpu
                .stream()
                .synchronize()
                .expect("case completion drain");
        }
        println!(
            "EXACT ordinal={ordinal} width={width} full logits, live cache, DSpark rings and sampled continuation"
        );
    }
    if compare.is_some() {
        println!(
            "PASS matrix/EP chunk identity; balanced comparison complete, no serving admission"
        );
    } else {
        println!(
            "PASS matrix/EP chunk identity; timings are single discovery rows, not balanced performance or serving admission"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_count, schedule};

    #[test]
    fn comparison_is_balanced_in_both_orders_with_six_rows_per_arm() {
        assert_eq!(schedule(None).unwrap(), [32, 64, 128, 256, 512, 32]);
        for width in [64, 128, 256, 512] {
            let runs = schedule(Some(&width.to_string())).unwrap();
            assert_eq!(runs.len(), 12);
            assert_eq!(runs.iter().filter(|&&w| w == 32).count(), 6);
            assert_eq!(runs.iter().filter(|&&w| w == width).count(), 6);
            for block in runs.chunks_exact(4) {
                assert_eq!(block, [32, width, width, 32]);
            }
        }
        for bad in ["32", "0", "63", "513", "x"] {
            assert!(schedule(Some(bad)).is_err());
        }
    }
    #[test]
    fn prompt_count_is_bounded_without_padding() {
        assert_eq!(parse_count(None), Ok(9900));
        for count in ["1025", "65536", "1048448"] {
            assert!(parse_count(Some(count)).is_ok());
        }
        for count in ["0", "1024", "1048449", "-1", "x", "18446744073709551616"] {
            assert!(parse_count(Some(count)).is_err());
        }
    }
}
