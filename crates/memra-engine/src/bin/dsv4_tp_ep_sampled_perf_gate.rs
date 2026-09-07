//! Plain sampled TP/EP performance smoke.
//!
//! This is deliberately a small same-topology repeat, not an ABBA campaign:
//! 256 source tokens are primed through the currently supported single-token
//! path, then 256 vendor-shape sampled tokens are generated. Prefill/prime and
//! decode are timed separately. No speculative path, PP arm, cache digest, or
//! hidden-state hash is inside either timed interval.

use memra_engine::dsv4_gpu::{Dsv4Gpu, Dsv4SampleCfg, dsv4_sample_row};
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::{
    path::Path,
    time::{Duration, Instant},
};

const PROMPT_TOKENS: usize = 256;
const OUTPUT_TOKENS: usize = 256;
const REPEATS: usize = 2;
const SOURCE_SHA256: &str = "f6e175a6f2588953568746fec0cd43fcd046405f74b5c71ce071fe7f37238ded";

#[derive(Clone, Copy, Debug, Default)]
struct Counters {
    rank_layer: [u64; 2],
    ep: u64,
    ar: u64,
    gu_m1: u64,
    gu_half2: u64,
    down_half2: u64,
    wo_a: u64,
    index_radix: u64,
}

fn counters(gpu: &Dsv4Gpu) -> Counters {
    Counters {
        rank_layer: gpu.tp_ep_rank_layer_calls(),
        ep: gpu.ep_calls(),
        ar: gpu.tp_ep_ar_dispatches(),
        gu_m1: memra_engine::moe_f16g_gu_m1_tc_dispatches(),
        gu_half2: memra_engine::moe_f16g_gu_half2_dispatches(),
        down_half2: memra_engine::moe_f16g_down_m1_half2_dispatches(),
        wo_a: gpu.dense_wo_a_grouped_dispatches(),
        index_radix: gpu.index_topk_radix_dispatches(),
    }
}

fn delta(after: Counters, before: Counters) -> Counters {
    Counters {
        rank_layer: [
            after.rank_layer[0] - before.rank_layer[0],
            after.rank_layer[1] - before.rank_layer[1],
        ],
        ep: after.ep - before.ep,
        ar: after.ar - before.ar,
        gu_m1: after.gu_m1 - before.gu_m1,
        gu_half2: after.gu_half2 - before.gu_half2,
        down_half2: after.down_half2 - before.down_half2,
        wo_a: after.wo_a - before.wo_a,
        index_radix: after.index_radix - before.index_radix,
    }
}

fn sha256_tokens(tokens: &[u32]) -> String {
    let mut h = Sha256::new();
    for &token in tokens {
        h.update(token.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

fn looped(tokens: &[u32]) -> bool {
    (1usize..=32).any(|width| {
        let length = width * 4usize.max(32usize.div_ceil(width));
        tokens.windows(length).any(|span| {
            span.chunks_exact(width)
                .all(|chunk| chunk == &span[..width])
        })
    })
}

fn drain(gpu: &Dsv4Gpu) {
    for stage in &gpu.stages {
        stage.gpu.stream().synchronize().expect("TP/EP perf drain");
    }
}

fn assert_engagement(gpu: &Dsv4Gpu, c: Counters, prime: usize, decode: usize) {
    let layers = gpu.topology().layers as u64;
    for (rank, &calls) in c.rank_layer.iter().enumerate() {
        assert_eq!(
            calls,
            (prime + decode) as u64 * layers,
            "rank {rank} layer-walk engagement"
        );
    }
    assert_eq!(
        c.ep,
        2 * (prime + decode) as u64 * layers,
        "local EP engagement"
    );
    assert_eq!(
        c.ar,
        (prime + decode) as u64 * layers,
        "one-shot AR engagement"
    );
    let local_steps = 2 * (prime + decode) as u64 * layers;
    assert_eq!(c.gu_m1, local_steps, "GU-M1 actual enqueues");
    assert_eq!(c.gu_half2, local_steps, "GU-half2 actual enqueues");
    assert_eq!(c.down_half2, local_steps, "down-half2 actual enqueues");
    assert_eq!(c.wo_a, local_steps, "grouped wo_a actual enqueues");
    // At prompt+output <= 512, the radix selector's N=2048 eligibility is
    // intentionally inert. The arm is still reported and checked as zero.
    assert_eq!(
        c.index_radix, 0,
        "short-context radix selector must be inert"
    );
}

#[derive(Debug)]
struct RunReceipt {
    repeat: usize,
    state_alloc: Duration,
    prime_wall: Duration,
    decode_wall: Duration,
    state_pos: usize,
    generated_sha256: String,
    looped: bool,
    counters_prime: Counters,
    counters_decode: Counters,
}

fn run_once(gpu: &Dsv4Gpu, prompt: &[u32], repeat: usize) -> RunReceipt {
    let alloc_start = Instant::now();
    let mut state = gpu
        .alloc_decode_state_for_transient(PROMPT_TOKENS + OUTPUT_TOKENS + 8, 1)
        .expect("TP/EP state");
    let state_alloc = alloc_start.elapsed();
    let before = counters(gpu);

    let prime_start = Instant::now();
    let mut row = gpu
        .prefill_with_cache_chunked(&prompt[..1], &mut state, 1)
        .expect("one-token TP/EP prime");
    for &token in &prompt[1..PROMPT_TOKENS] {
        row = gpu
            .decode_step(token, &mut state)
            .expect("TP/EP prompt prime");
    }
    drain(gpu);
    let prime_wall = prime_start.elapsed();
    assert_eq!(state.pos, PROMPT_TOKENS, "prime position");
    let after_prime = counters(gpu);
    let counters_prime = delta(after_prime, before);
    assert_engagement(gpu, counters_prime, PROMPT_TOKENS, 0);

    let cfg = Dsv4SampleCfg {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        seed: 20260907,
    };
    let mut generated = Vec::with_capacity(OUTPUT_TOKENS);
    let decode_start = Instant::now();
    for _ in 0..OUTPUT_TOKENS {
        let token = dsv4_sample_row(&row, state.pos, &cfg).expect("sample");
        generated.push(token);
        row = gpu
            .decode_step(token, &mut state)
            .expect("TP/EP sampled decode");
    }
    drain(gpu);
    let decode_wall = decode_start.elapsed();
    assert_eq!(
        state.pos,
        PROMPT_TOKENS + OUTPUT_TOKENS,
        "sampled decode position"
    );
    let counters_decode = delta(counters(gpu), after_prime);
    assert_engagement(gpu, counters_decode, 0, OUTPUT_TOKENS);

    let generated_sha256 = sha256_tokens(&generated);
    let is_looped = looped(&generated);
    let decode_tok_s = OUTPUT_TOKENS as f64 / decode_wall.as_secs_f64();
    let prime_tok_s = PROMPT_TOKENS as f64 / prime_wall.as_secs_f64();
    let eligible = !is_looped;
    println!(
        "MEASURE {{\"repeat\":{repeat},\"prompt_tokens\":{PROMPT_TOKENS},\"output_tokens\":{OUTPUT_TOKENS},\"state_alloc_ns\":{},\"prime_wall_ns\":{},\"decode_wall_ns\":{},\"prime_tok_s\":{prime_tok_s:.6},\"decode_tok_s\":{decode_tok_s:.6},\"eligible\":{eligible},\"looped\":{is_looped},\"state_pos\":{},\"generated_sha256\":\"{generated_sha256}\",\"rank_layer_calls\":[{},{}],\"ep_calls\":{},\"ar_dispatches\":{},\"gu_m1_calls\":{},\"gu_half2_calls\":{},\"down_half2_calls\":{},\"wo_a_calls\":{},\"index_radix_calls\":{},\"speculative\":false,\"pp_timing\":false,\"cache_hash_in_timing\":false,\"hidden_hash_in_timing\":false}}",
        state_alloc.as_nanos(),
        prime_wall.as_nanos(),
        decode_wall.as_nanos(),
        state.pos,
        counters_decode.rank_layer[0],
        counters_decode.rank_layer[1],
        counters_decode.ep,
        counters_decode.ar,
        counters_decode.gu_m1,
        counters_decode.gu_half2,
        counters_decode.down_half2,
        counters_decode.wo_a,
        counters_decode.index_radix,
    );
    // Keep the finite check outside the timed interval.
    assert!(
        row.iter().all(|value| value.is_finite()),
        "final logits finite"
    );
    RunReceipt {
        repeat,
        state_alloc,
        prime_wall,
        decode_wall,
        state_pos: state.pos,
        generated_sha256,
        looped: is_looped,
        counters_prime,
        counters_decode,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(
        args.len(),
        3,
        "usage: dsv4_tp_ep_sampled_perf_gate <model-dir> <real-source.txt>"
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
        "sampled plain TP/EP gate refuses DSpark"
    );
    let dir = Path::new(&args[1]);
    let source = std::fs::read_to_string(&args[2]).expect("source");
    assert_eq!(
        format!("{:x}", Sha256::digest(source.as_bytes())),
        SOURCE_SHA256,
        "pinned source"
    );
    let tokenizer = Tokenizer::from_hf_dir(dir).expect("tokenizer");
    let prompt = tokenizer.encode(
        &format!("Review this inference engine source:\n\n{source}"),
        true,
    );
    assert!(prompt.len() >= PROMPT_TOKENS);

    Dsv4Gpu::set_tp_ep_topology_for_gate(true);
    println!(
        "PROTOCOL {{\"plain_only\":true,\"sampled\":true,\"topology\":\"tp_ep_all_layers\",\"prompt_tokens\":{PROMPT_TOKENS},\"output_tokens\":{OUTPUT_TOKENS},\"repeats\":{REPEATS},\"temperature\":1.0,\"top_p\":1.0,\"top_k\":0,\"seed\":20260907,\"source_sha256\":\"{SOURCE_SHA256}\",\"speculative\":false,\"pp_timing\":false,\"cache_hash_in_timing\":false}}"
    );
    let gpu = Dsv4Gpu::load(
        dir,
        &[0, 1],
        ActQuantVariant::RefFp8Round,
        PROMPT_TOKENS + OUTPUT_TOKENS + 32,
    )
    .expect("TP/EP model");
    assert!(gpu.topology().is_tp_ep(), "no PP fallback");
    gpu.set_grouped_route_validation_for_gate(false);
    gpu.set_grouped_mirror_validation_for_gate(false);
    gpu.set_grouped_gu_fuse_for_gate(true);
    gpu.set_grouped_m1_tc_for_gate(true);
    memra_engine::set_moe_f16g_gu_half2_for_gate(true);
    memra_engine::set_moe_f16g_down_m1_half2_for_gate(true);
    gpu.set_dense_wo_a_grouped_for_gate(true);
    gpu.set_index_topk_radix_for_gate(true);

    let first = run_once(&gpu, &prompt, 0);
    let second = run_once(&gpu, &prompt, 1);
    assert_eq!(
        first.generated_sha256, second.generated_sha256,
        "same-TP sampled repeat stream"
    );
    assert_eq!(first.state_pos, second.state_pos);
    println!("PASS sampled TP/EP internal repeat; loop exclusion remains per-row eligibility");
    Dsv4Gpu::set_tp_ep_topology_for_gate(false);
}
