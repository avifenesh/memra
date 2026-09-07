//! Plain sampled TP/EP performance smoke.
//!
//! This is deliberately a small same-topology repeat, not an ABBA campaign:
//! 256 source tokens are primed through the currently supported single-token
//! path, then 256 vendor-shape sampled tokens are generated. Prefill/prime and
//! decode are timed separately. No speculative path, PP arm, cache digest, or
//! hidden-state hash is inside either timed interval.

use memra_engine::dsv4_gpu::{
    Dsv4Gpu, Dsv4Phase, Dsv4SampleCfg, Dsv4SamplerOrder, dsv4_prof_on, dsv4_sample_row,
    dsv4_sampler_order,
};
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
const ATTENTION_REPEATS: usize = 5;
const SOURCE_SHA256: &str = "f6e175a6f2588953568746fec0cd43fcd046405f74b5c71ce071fe7f37238ded";

#[derive(Clone, Copy, Debug, Default)]
struct Counters {
    rank_layer: [u64; 2],
    ep: u64,
    ar: u64,
    attention_rank: [u64; 2],
    attention_ar: u64,
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
        attention_rank: gpu.attention_tp_rank_calls(),
        attention_ar: gpu.attention_tp_ar_calls(),
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
        attention_rank: std::array::from_fn(|rank| {
            after.attention_rank[rank] - before.attention_rank[rank]
        }),
        attention_ar: after.attention_ar - before.attention_ar,
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

fn sha256_f32(values: &[f32]) -> String {
    let mut h = Sha256::new();
    for &value in values {
        h.update(value.to_bits().to_le_bytes());
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

/// The sampled decode wall includes both the sampler and the following forward.
/// Keep this boundary in one helper so a future timing edit cannot silently turn
/// the headline into a forward-only number.
fn timed_sampled_decode<F, G>(steps: usize, mut step: F, finish: G) -> (Duration, usize)
where
    F: FnMut(usize) -> bool,
    G: FnOnce(),
{
    let start = Instant::now();
    let mut completed = 0usize;
    for index in 0..steps {
        if !step(index) {
            break;
        }
        completed += 1;
    }
    finish();
    (start.elapsed(), completed)
}

#[cfg(test)]
mod timing_contract_tests {
    use super::timed_sampled_decode;
    use std::time::Duration;

    #[test]
    fn sampled_wall_includes_injected_sampler_delay() {
        let (elapsed, completed) = timed_sampled_decode(
            1,
            |_| {
                std::thread::sleep(Duration::from_millis(5));
                true
            },
            || {},
        );
        assert_eq!(completed, 1);
        assert!(elapsed >= Duration::from_millis(4));
    }

    #[test]
    fn sampled_wall_includes_injected_finish_drain_delay() {
        let (elapsed, completed) = timed_sampled_decode(
            1,
            |_| true,
            || {
                std::thread::sleep(Duration::from_millis(5));
            },
        );
        assert_eq!(completed, 1);
        assert!(elapsed >= Duration::from_millis(4));
    }

    #[test]
    fn sampled_wall_preserves_early_eos_count() {
        let (elapsed, completed) = timed_sampled_decode(4, |index| index < 2, || {});
        assert_eq!(completed, 2);
        assert!(elapsed < Duration::from_millis(100));
    }
}

fn assert_engagement(gpu: &Dsv4Gpu, c: Counters, prime: usize, decode: usize) {
    let layers = gpu.topology().layers as u64;
    let attention_mode = gpu.attention_tp_geometry().is_some();
    let attention_steps = u64::from(attention_mode) * (prime + decode) as u64 * layers;
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
        (prime + decode) as u64 * layers + attention_steps,
        "one-shot AR engagement"
    );
    assert_eq!(
        c.attention_rank, [attention_steps; 2],
        "actual attention rank producers"
    );
    assert_eq!(
        c.attention_ar, attention_steps,
        "actual attention reductions"
    );
    let local_steps = 2 * (prime + decode) as u64 * layers;
    assert_eq!(c.gu_m1, local_steps, "GU-M1 actual enqueues");
    assert_eq!(c.gu_half2, local_steps, "GU-half2 actual enqueues");
    assert_eq!(c.down_half2, local_steps, "down-half2 actual enqueues");
    assert_eq!(
        c.wo_a,
        if attention_mode { 0 } else { local_steps },
        "attention TP uses per-group wo_a; replicated attention uses qualified grouped wo_a"
    );
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
    generated_tokens: usize,
    forward_calls: usize,
    eos: bool,
    eligible: bool,
    final_logits_sha256: String,
    final_cache_digest: [u64; 2],
    final_hidden_digest: [u64; 2],
    attention_join_sha256: Option<String>,
    ar_refusals: [i32; 2],
    looped: bool,
    counters_prime: Counters,
    counters_decode: Counters,
}

fn run_once(gpu: &Dsv4Gpu, prompt: &[u32], tokenizer: &Tokenizer, repeat: usize) -> RunReceipt {
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
    let prime_ar_refusals = gpu
        .tp_ep_ar_refusal_words()
        .expect("prime AR refusal words");
    assert_eq!(prime_ar_refusals, [0, 0], "prime AR refusal");

    let cfg = Dsv4SampleCfg {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        seed: 20260907,
    };
    let mut generated = Vec::with_capacity(OUTPUT_TOKENS);
    let mut eos = false;
    let profiled = dsv4_prof_on();
    let mut decode_phase = None;
    // The headline clock includes CPU sampling, every forward, and the final drain.
    // A sum of decode_step durations would be forward-only and is not this metric.
    let (decode_wall, forward_calls) = timed_sampled_decode(
        OUTPUT_TOKENS,
        |_| {
            if profiled && generated.len() == 32 {
                drain(gpu);
                decode_phase = Dsv4Phase::new("TP_EP_DECODE\0", None);
            }
            let token = dsv4_sample_row(&row, state.pos, &cfg).expect("sample");
            if token == tokenizer.eos_id() {
                eos = true;
                return false;
            }
            generated.push(token);
            row = gpu
                .decode_step(token, &mut state)
                .expect("TP/EP sampled decode");
            if profiled && generated.len() == 64 {
                drain(gpu);
                drop(decode_phase.take());
            }
            true
        },
        || drain(gpu),
    );
    drop(decode_phase);
    assert_eq!(
        state.pos,
        PROMPT_TOKENS + generated.len(),
        "sampled decode position"
    );
    assert_eq!(forward_calls, generated.len(), "sampled forward count");
    let counters_decode = delta(counters(gpu), after_prime);
    assert_engagement(gpu, counters_decode, 0, generated.len());
    let ar_refusals = gpu
        .tp_ep_ar_refusal_words()
        .expect("decode AR refusal words");
    assert_eq!(ar_refusals, [0, 0], "decode AR refusal");
    assert!(
        row.iter().all(|value| value.is_finite()),
        "final logits finite"
    );

    let generated_sha256 = sha256_tokens(&generated);
    let is_looped = looped(&generated);
    let decode_tok_s = generated.len() as f64 / decode_wall.as_secs_f64();
    let prime_tok_s = PROMPT_TOKENS as f64 / prime_wall.as_secs_f64();
    let eligible = !profiled && !eos && generated.len() == OUTPUT_TOKENS && !is_looped;
    let headline_decode_tok_s = if eligible {
        format!("{decode_tok_s:.6}")
    } else {
        "null".to_string()
    };
    // All identity material is collected after timing and after the refusal
    // checks. These are consistency receipts, not an oracle-equivalence gate.
    let final_logits_sha256 = sha256_f32(&row);
    let final_cache_digest = gpu
        .tp_ep_cache_digest_for_gate(&state)
        .expect("final cache digest");
    let final_hidden_digest = gpu
        .tp_ep_hidden_digest_for_gate(&state)
        .expect("final hidden digest");
    assert_eq!(
        final_cache_digest[0], final_cache_digest[1],
        "final cache rank symmetry"
    );
    assert_eq!(
        final_hidden_digest[0], final_hidden_digest[1],
        "final hidden rank symmetry"
    );
    let attention_mode = gpu.attention_tp_geometry().is_some();
    let attention_join_sha256 = if attention_mode {
        let snapshot = gpu
            .attention_tp_last_join_for_gate(&state)
            .expect("actual final attention join");
        let hidden = gpu.attention_tp_geometry().unwrap().hidden;
        for plane in snapshot.partials.iter().chain(snapshot.joined.iter()) {
            assert_eq!(plane.len(), hidden);
            assert!(plane.iter().all(|value| value.is_finite()));
        }
        for (column, (&rank0, &rank1)) in snapshot.partials[0]
            .iter()
            .zip(&snapshot.partials[1])
            .enumerate()
        {
            let expected = rank0 + rank1;
            assert!(expected.is_finite());
            for joined in &snapshot.joined {
                assert_eq!(
                    joined[column].to_bits(),
                    expected.to_bits(),
                    "actual attention GPU sum vs CPU f32 at {column}"
                );
            }
        }
        Some(sha256_f32(&snapshot.joined[0]))
    } else {
        None
    };
    let attention_join_json = attention_join_sha256
        .as_ref()
        .map_or("null".to_string(), |value| format!("\"{value}\""));
    println!("TOKENS {{\"repeat\":{repeat},\"ids\":{generated:?}}}");
    println!(
        "OUTPUT_TEXT repeat={repeat} text={:?}",
        tokenizer.decode(&generated)
    );
    println!("PROFILE repeat={repeat} enabled={profiled} window_start=32 window_end=64");
    println!(
        "MEASURE {{\"repeat\":{repeat},\"prompt_tokens\":{PROMPT_TOKENS},\"requested_output_tokens\":{OUTPUT_TOKENS},\"generated_tokens\":{},\"forward_calls\":{},\"eos\":{eos},\"state_alloc_ns\":{},\"prime_wall_ns\":{},\"decode_wall_ns\":{},\"timing_scope\":\"sample_plus_forward_envelope\",\"sampling_in_timing\":true,\"prime_tok_s\":{prime_tok_s:.6},\"decode_tok_s\":{decode_tok_s:.6},\"headline_decode_tok_s\":{headline_decode_tok_s},\"eligible\":{eligible},\"looped\":{is_looped},\"state_pos\":{},\"generated_sha256\":\"{generated_sha256}\",\"final_logits_sha256\":\"{final_logits_sha256}\",\"final_cache_digest\":[{},{}],\"final_hidden_digest\":[{},{}],\"attention_tp\":{attention_mode},\"attention_join_sha256\":{attention_join_json},\"attention_rank_calls\":[{},{}],\"attention_ar_calls\":{},\"ar_refusals\":[{},{}],\"rank_layer_calls\":[{},{}],\"ep_calls\":{},\"ar_dispatches\":{},\"gu_m1_calls\":{},\"gu_half2_calls\":{},\"down_half2_calls\":{},\"wo_a_calls\":{},\"index_radix_calls\":{},\"speculative\":false,\"pp_timing\":false,\"cache_hash_in_timing\":false,\"hidden_hash_in_timing\":false}}",
        generated.len(),
        generated.len(),
        state_alloc.as_nanos(),
        prime_wall.as_nanos(),
        decode_wall.as_nanos(),
        state.pos,
        final_cache_digest[0],
        final_cache_digest[1],
        final_hidden_digest[0],
        final_hidden_digest[1],
        counters_decode.attention_rank[0],
        counters_decode.attention_rank[1],
        counters_decode.attention_ar,
        ar_refusals[0],
        ar_refusals[1],
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
    RunReceipt {
        repeat,
        state_alloc,
        prime_wall,
        decode_wall,
        state_pos: state.pos,
        generated_sha256,
        generated_tokens: generated.len(),
        forward_calls: generated.len(),
        eos,
        eligible,
        final_logits_sha256,
        final_cache_digest,
        final_hidden_digest,
        attention_join_sha256,
        ar_refusals,
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
    let attention_mode = match std::env::var("MEMRA_DSV4_ATTENTION_TP_GATE").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("0") => false,
        Ok("1") => true,
        _ => panic!("MEMRA_DSV4_ATTENTION_TP_GATE requires 0 or 1"),
    };
    let sampler = dsv4_sampler_order().expect("explicit sampler configuration");
    if attention_mode {
        assert_eq!(
            sampler,
            Dsv4SamplerOrder::Comparison,
            "first attention-TP performance receipt pins Comparison sampling"
        );
        assert!(
            !dsv4_prof_on(),
            "attention TP performance refuses profiled timing"
        );
    }
    let sampler_name = match sampler {
        Dsv4SamplerOrder::Comparison => "comparison",
        Dsv4SamplerOrder::Radix => "radix",
    };
    let repeats = if attention_mode {
        ATTENTION_REPEATS
    } else {
        REPEATS
    };
    let numeric_class = if attention_mode {
        memra_engine::dsv4_attention_tp::ATTENTION_TP_NUMERIC_CLASS
    } else {
        memra_engine::dsv4_gpu::TP_EP_RANK_ORDER_NUMERIC_CLASS
    };
    assert_ne!(
        std::env::var("MEMRA_DSV4_ROUND_PROFILE").as_deref(),
        Ok("1"),
        "use NVTX-only profiling; sync-bracketed timings are not admitted by this gate"
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
    Dsv4Gpu::set_attention_tp_for_gate(attention_mode);
    println!("NUMERIC_CLASS {numeric_class}");
    println!(
        "PROTOCOL {{\"plain_only\":true,\"sampled\":true,\"topology\":\"tp_ep_all_layers\",\"attention_tp\":{attention_mode},\"prompt_tokens\":{PROMPT_TOKENS},\"output_tokens\":{OUTPUT_TOKENS},\"repeats\":{repeats},\"temperature\":1.0,\"top_p\":1.0,\"top_k\":0,\"seed\":20260907,\"sampler_order\":\"{sampler_name}\",\"timing_scope\":\"sample_plus_forward_envelope\",\"sampling_in_timing\":true,\"source_sha256\":\"{SOURCE_SHA256}\",\"speculative\":false,\"pp_timing\":false,\"cache_hash_in_timing\":false}}"
    );
    let gpu = Dsv4Gpu::load(
        dir,
        &[0, 1],
        ActQuantVariant::RefFp8Round,
        PROMPT_TOKENS + OUTPUT_TOKENS + 32,
    )
    .expect("TP/EP model");
    assert!(gpu.topology().is_tp_ep(), "no PP fallback");
    assert_eq!(
        gpu.attention_tp_geometry().is_some(),
        attention_mode,
        "no attention fallback"
    );
    gpu.set_grouped_route_validation_for_gate(false);
    gpu.set_grouped_mirror_validation_for_gate(false);
    gpu.set_grouped_gu_fuse_for_gate(true);
    gpu.set_grouped_m1_tc_for_gate(true);
    memra_engine::set_moe_f16g_gu_m1_tc_for_gate(true);
    memra_engine::set_moe_f16g_gu_half2_for_gate(true);
    memra_engine::set_moe_f16g_down_m1_half2_for_gate(true);
    gpu.set_dense_wo_a_grouped_for_gate(!attention_mode);
    gpu.set_index_topk_radix_for_gate(true);

    let receipts: Vec<_> = (0..repeats)
        .map(|repeat| run_once(&gpu, &prompt, &tokenizer, repeat))
        .collect();
    let first = &receipts[0];
    for receipt in &receipts {
        assert_eq!(
            first.generated_sha256, receipt.generated_sha256,
            "same-program sampled repeat stream"
        );
        assert_eq!(first.state_pos, receipt.state_pos);
        assert_eq!(first.generated_tokens, receipt.generated_tokens);
        assert_eq!(first.forward_calls, receipt.forward_calls);
        assert_eq!(first.eos, receipt.eos);
        assert_eq!(first.final_logits_sha256, receipt.final_logits_sha256);
        assert_eq!(first.final_cache_digest, receipt.final_cache_digest);
        assert_eq!(first.final_hidden_digest, receipt.final_hidden_digest);
        assert_eq!(first.attention_join_sha256, receipt.attention_join_sha256);
        assert_eq!(receipt.ar_refusals, [0, 0]);
        println!(
            "REPEAT repeat={} state_alloc_ns={} prime_wall_ns={} decode_wall_ns={} state_pos={} generated_tokens={} forward_calls={} eos={} looped={} eligible={} prime_counters={:?} decode_counters={:?}",
            receipt.repeat,
            receipt.state_alloc.as_nanos(),
            receipt.prime_wall.as_nanos(),
            receipt.decode_wall.as_nanos(),
            receipt.state_pos,
            receipt.generated_tokens,
            receipt.forward_calls,
            receipt.eos,
            receipt.looped,
            receipt.eligible,
            receipt.counters_prime,
            receipt.counters_decode,
        );
    }
    if attention_mode {
        assert!(
            receipts.iter().all(|receipt| receipt.eligible),
            "all five sampled attention rows must be eligible; no reroll or forward-only substitute"
        );
        let total_tokens: usize = receipts
            .iter()
            .map(|receipt| receipt.generated_tokens)
            .sum();
        let total_wall_ns: u128 = receipts
            .iter()
            .map(|receipt| receipt.decode_wall.as_nanos())
            .sum();
        let pooled_tok_s = total_tokens as f64 * 1e9 / total_wall_ns as f64;
        println!(
            "SUMMARY {{\"repeats\":{repeats},\"eligible_repeats\":{repeats},\"generated_tokens\":{total_tokens},\"decode_wall_ns\":{total_wall_ns},\"sampled_envelope_tok_s\":{pooled_tok_s:.6},\"timing_scope\":\"sample_plus_forward_envelope\",\"sampler_order\":\"comparison\",\"paired_control\":false,\"speculative\":false}}"
        );
    }
    if attention_mode {
        println!(
            "PASS sampled attention TP2; repeats={repeats} eligible={repeats} timing_scope=sample_plus_forward_envelope sampler=comparison"
        );
    } else {
        println!(
            "PASS sampled TP/EP internal repeat; eligible_first={} eligible_second={} loop exclusion remains per-row eligibility",
            first.eligible, receipts[1].eligible
        );
    }
    Dsv4Gpu::set_attention_tp_for_gate(false);
    Dsv4Gpu::set_tp_ep_topology_for_gate(false);
}
