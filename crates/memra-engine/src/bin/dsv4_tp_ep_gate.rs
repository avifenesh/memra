//! Narrow plain-only gate for the all-layer DSV4 TP/EP walk.
//!
//! This gate intentionally exercises only the currently implemented shape:
//! one-token prime followed by teacher-forced single-token continuation. It
//! arms TP/EP before loading, refuses DSpark/MTP, and requires both replicated
//! ranks to execute every trunk layer. It is a correctness/engagement gate,
//! not a throughput benchmark and has no PP comparison arm.

use memra_engine::dsv4_gpu::{Dsv4Gpu, TP_EP_RANK_ORDER_NUMERIC_CLASS};
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::path::Path;

const CONTINUATION_TOKENS: usize = 3;
const PINNED_SOURCE_SHA256: &str =
    "f6e175a6f2588953568746fec0cd43fcd046405f74b5c71ce071fe7f37238ded";

#[derive(Clone, Debug, PartialEq, Eq)]
struct Receipt {
    source_sha256: String,
    output_sha256: String,
    position: usize,
    positions: Vec<usize>,
    rank_layer_calls: [u64; 2],
    cache_digests: Vec<[u64; 2]>,
    hidden_digests: Vec<[u64; 2]>,
    ep_calls: u64,
    ar_dispatches: u64,
    ar_refusals: [i32; 2],
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn update_f32(hash: &mut Sha256, values: &[f32]) {
    assert!(
        values.iter().all(|value| value.is_finite()),
        "TP/EP output contains a non-finite value"
    );
    for value in values {
        hash.update(value.to_bits().to_le_bytes());
    }
}

fn drain(gpu: &Dsv4Gpu) {
    for stage in &gpu.stages {
        stage.gpu.stream().synchronize().expect("TP/EP gate drain");
    }
}

fn run_once(gpu: &Dsv4Gpu, tokens: &[u32], source_sha256: &str) -> Receipt {
    assert!(tokens.len() >= CONTINUATION_TOKENS + 1);
    let trunk_layers = gpu.topology().layers as u64;
    let rank_layer_before = gpu.tp_ep_rank_layer_calls();
    let ar_before = gpu.tp_ep_ar_dispatches();
    let ep_before = gpu.ep_calls();
    let steps = CONTINUATION_TOKENS + 1;
    let mut state = gpu
        .alloc_decode_state_for_transient(steps + 8, 1)
        .expect("TP/EP decode state");
    let mut output_hash = Sha256::new();
    let mut cache_digests = Vec::with_capacity(CONTINUATION_TOKENS + 1);
    let mut hidden_digests = Vec::with_capacity(CONTINUATION_TOKENS + 1);
    let mut positions = Vec::with_capacity(steps);

    // The supported prefill shape is deliberately one token. The following
    // source tokens exercise the actual continuation API, not a synthetic
    // kernel-only probe.
    let row = gpu
        .prefill_with_cache_chunked(&tokens[..1], &mut state, 1)
        .expect("TP/EP one-token prime");
    assert_eq!(state.pos, 1, "TP/EP prime must commit one token");
    positions.push(state.pos);
    update_f32(&mut output_hash, &row);
    let digest = gpu
        .tp_ep_cache_digest_for_gate(&state)
        .expect("TP/EP prime cache digest");
    assert_eq!(digest[0], digest[1], "prime cache rank symmetry");
    cache_digests.push(digest);
    let hidden = gpu
        .tp_ep_hidden_digest_for_gate(&state)
        .expect("TP/EP prime hidden digest");
    assert_eq!(hidden[0], hidden[1], "prime hidden rank symmetry");
    hidden_digests.push(hidden);
    for &token in &tokens[1..=CONTINUATION_TOKENS] {
        let row = gpu
            .decode_step(token, &mut state)
            .expect("TP/EP continuation");
        let expected_position = positions.len() + 1;
        assert_eq!(
            state.pos, expected_position,
            "TP/EP continuation position must advance after commit"
        );
        positions.push(state.pos);
        update_f32(&mut output_hash, &row);
        let digest = gpu
            .tp_ep_cache_digest_for_gate(&state)
            .expect("TP/EP continuation cache digest");
        assert_eq!(digest[0], digest[1], "continuation cache rank symmetry");
        cache_digests.push(digest);
        let hidden = gpu
            .tp_ep_hidden_digest_for_gate(&state)
            .expect("TP/EP continuation hidden digest");
        assert_eq!(hidden[0], hidden[1], "continuation hidden rank symmetry");
        hidden_digests.push(hidden);
    }
    assert!(
        cache_digests
            .windows(2)
            .all(|pair| pair[0][0] != pair[1][0] && pair[0][1] != pair[1][1]),
        "persistent cache data must progress on both ranks after every token"
    );
    let final_position = state.pos;
    drop(state);
    drain(gpu);

    assert_eq!(positions.len(), steps);
    let rank_layer_after = gpu.tp_ep_rank_layer_calls();
    let rank_layer_calls = [
        rank_layer_after[0] - rank_layer_before[0],
        rank_layer_after[1] - rank_layer_before[1],
    ];
    let steps = steps as u64;
    assert_eq!(
        rank_layer_calls,
        [steps * trunk_layers, steps * trunk_layers],
        "both TP/EP ranks must execute every trunk layer on every token"
    );
    let ep_calls = gpu.ep_calls() - ep_before;
    assert_eq!(
        ep_calls,
        2 * steps * trunk_layers,
        "local expert execution must engage once per rank/layer/token"
    );
    let ar_dispatches = gpu.tp_ep_ar_dispatches() - ar_before;
    assert_eq!(
        ar_dispatches,
        steps * trunk_layers,
        "one named rank-order reduction per layer/token"
    );
    let ar_refusals = gpu
        .tp_ep_ar_refusal_words()
        .expect("TP/EP AR refusal words");
    assert_eq!(
        ar_refusals,
        [0, 0],
        "one-shot AR must not refuse on either rank"
    );
    Receipt {
        source_sha256: source_sha256.to_owned(),
        output_sha256: sha256_bytes(&output_hash.finalize()),
        position: final_position,
        positions,
        rank_layer_calls,
        cache_digests,
        hidden_digests,
        ep_calls,
        ar_dispatches,
        ar_refusals,
    }
}

fn verify_refusal_boundary(gpu: &Dsv4Gpu, tokens: &[u32]) {
    assert!(tokens.len() >= 128, "refusal gate needs the C128 boundary");
    for (rank, code) in [(0usize, 40043), (1usize, 40044)] {
        for prime_tokens in [1usize, 3, 127] {
            let mut state = gpu
                .alloc_decode_state_for_transient(prime_tokens + 8, 1)
                .expect("refusal gate state");
            gpu.prefill_with_cache_chunked(&tokens[..1], &mut state, 1)
                .expect("refusal gate prime");
            for &token in &tokens[1..prime_tokens] {
                gpu.decode_step(token, &mut state)
                    .expect("refusal gate teacher-forced prime continuation");
            }
            let position = state.pos;
            let cache_before = gpu
                .tp_ep_cache_digest_for_gate(&state)
                .expect("refusal gate initial cache");
            let mut words = [0, 0];
            words[rank] = code;
            gpu.set_tp_ep_ar_refusal_words_for_gate(words)
                .expect("inject sticky refusal");
            let error = gpu
                .decode_step(tokens[prime_tokens], &mut state)
                .err()
                .expect("sticky device refusal must reject token");
            assert!(error.contains("one-shot reduction refused"), "{error}");
            assert!(error.contains(&code.to_string()), "{error}");
            assert_eq!(
                state.pos, position,
                "refused token must not advance position"
            );
            assert_eq!(
                gpu.tp_ep_cache_digest_for_gate(&state)
                    .expect("refusal gate final cache"),
                cache_before,
                "refusal must be observed before either persistent cache commits"
            );
            // Clearing the diagnostic word does not rehabilitate a failed request.
            gpu.set_tp_ep_ar_refusal_words_for_gate([0, 0])
                .expect("clear completed red arm");
            let calls = gpu.tp_ep_rank_layer_calls();
            let retry = gpu
                .decode_step(tokens[prime_tokens], &mut state)
                .err()
                .expect("failed request must remain unusable");
            assert!(retry.contains("unfinished transaction"), "{retry}");
            assert_eq!(
                gpu.tp_ep_rank_layer_calls(),
                calls,
                "retry must not enqueue layers"
            );
            assert_eq!(
                state.pos, position,
                "failed retry must not advance position"
            );
            println!(
                "REFUSAL_GATE rank={rank} code={code} position={position} cache_unchanged=true retry_refused=true"
            );
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(
        args.len(),
        3,
        "usage: dsv4_tp_ep_gate <model-dir> <real-source.txt>"
    );
    for (name, expected) in [
        ("MEMRA_DSV4_DECODE_PATH", "device"),
        ("MEMRA_DSV4_EXPERT_ARM", "native"),
        ("MEMRA_DSV4_DENSE_ARM", "fp8"),
        ("MEMRA_DSV4_EP", "pair"),
        ("MEMRA_DSV4_MOE_PROGRAM", "matrix"),
        ("MEMRA_DSV4_GROUPED_ROUTE", "device"),
        ("MEMRA_DSV4_VERIFY_TOPK", "device"),
        ("MEMRA_DSV4_PREFILL_MOE", "reference"),
    ] {
        assert_eq!(
            std::env::var(name).as_deref(),
            Ok(expected),
            "requires {name}={expected}"
        );
    }
    assert!(
        matches!(
            std::env::var("MEMRA_DSV4_DRAFTER").as_deref(),
            Err(_) | Ok("") | Ok("off")
        ),
        "plain-only TP/EP gate refuses MEMRA_DSV4_DRAFTER=dspark"
    );

    let dir = Path::new(&args[1]);
    let source = std::fs::read_to_string(&args[2]).expect("source");
    let source_sha256 = sha256_bytes(source.as_bytes());
    assert_eq!(
        source_sha256, PINNED_SOURCE_SHA256,
        "pinned real-source tape"
    );
    let tokenizer = Tokenizer::from_hf_dir(dir).expect("tokenizer");
    let prompt = tokenizer.encode(
        &format!("Review this inference engine source:\n\n{source}"),
        true,
    );
    assert!(
        prompt.len() >= CONTINUATION_TOKENS + 1,
        "source must provide enough real tokens"
    );

    Dsv4Gpu::set_tp_ep_topology_for_gate(true);
    println!(
        "PROTOCOL {{\"plain_only\":true,\"topology\":\"tp_ep_all_layers\",\"numeric_class\":\"{TP_EP_RANK_ORDER_NUMERIC_CLASS}\",\"prime_tokens\":1,\"continuation_tokens\":{CONTINUATION_TOKENS},\"source_sha256\":\"{source_sha256}\",\"dspark\":false}}"
    );
    let mut gpu = Dsv4Gpu::load(dir, &[0, 1], ActQuantVariant::RefFp8Round, 256)
        .expect("plain-only TP/EP load");
    assert!(gpu.topology().is_tp_ep(), "no silent PP fallback");
    assert!(
        gpu.dspark.is_none(),
        "DSpark must not be resident in this gate"
    );
    assert!(gpu.mtp.is_none(), "MTP must not be resident in this gate");
    gpu.set_grouped_route_validation_for_gate(false);
    gpu.set_grouped_mirror_validation_for_gate(false);
    gpu.set_grouped_gu_fuse_for_gate(true);
    gpu.set_grouped_m1_tc_for_gate(true);
    assert_eq!(
        gpu.tp_ep_ar_refusal_words().expect("initial AR words"),
        [0, 0]
    );

    let first = run_once(&gpu, &prompt, &source_sha256);
    let second = run_once(&gpu, &prompt, &source_sha256);
    assert_eq!(
        first, second,
        "repeated plain TP/EP tape must be deterministic"
    );
    println!("RECEIPT {first:?}");
    verify_refusal_boundary(&gpu, &prompt);
    println!(
        "PASS plain-only all-layer TP/EP ranks={} layers={} numeric_class={} no_pp_fallback=true deterministic=true internal_consistency=true refusal_boundary=true oracle_equivalence=false",
        gpu.topology().world,
        gpu.topology().layers,
        TP_EP_RANK_ORDER_NUMERIC_CLASS
    );
    Dsv4Gpu::set_tp_ep_topology_for_gate(false);
}
