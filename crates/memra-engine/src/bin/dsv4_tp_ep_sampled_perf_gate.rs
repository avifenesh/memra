//! Plain sampled TP/EP performance smoke.
//!
//! Default mode is a small same-topology repeat. `--sampler-abba` runs 10
//! interleaved host/device/device/host cycles with full final-state identity.
//! 256 source tokens are primed through the currently supported single-token
//! path, then 256 vendor-shape sampled tokens are generated. Prefill/prime and
//! decode are timed separately. No speculative path, PP arm, cache digest, or
//! hidden-state hash is inside either timed interval.

use memra_engine::dsv4_gpu::{
    Dsv4Gpu, Dsv4Phase, Dsv4SampleCfg, Dsv4SamplerOrder, dsv4_prof_on, dsv4_sample_row,
    dsv4_sampler_order,
};
use memra_engine::dsv4_sampler::{Dsv4DeviceSampler, Dsv4Sampler, dsv4_sampler};
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::{
    path::Path,
    time::{Duration, Instant},
};

#[path = "../dsv4_full_token_replay_gate.rs"]
mod full_token_replay;

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
    splitk_gu: u64,
    splitk_down: u64,
    gu_half2: u64,
    down_half2: u64,
    wo_a: u64,
    index_radix: u64,
    small_launches: [u64; 2],
}

fn counters(gpu: &Dsv4Gpu) -> Counters {
    Counters {
        rank_layer: gpu.tp_ep_rank_layer_calls(),
        ep: gpu.ep_calls(),
        ar: gpu.tp_ep_ar_dispatches(),
        attention_rank: gpu.attention_tp_rank_calls(),
        attention_ar: gpu.attention_tp_ar_calls(),
        splitk_gu: memra_engine::MOE_M1_SPLITK_GU_DISPATCHES
            .load(std::sync::atomic::Ordering::Relaxed),
        splitk_down: memra_engine::MOE_M1_SPLITK_DOWN_DISPATCHES
            .load(std::sync::atomic::Ordering::Relaxed),
        gu_m1: memra_engine::moe_f16g_gu_m1_tc_dispatches(),
        gu_half2: memra_engine::moe_f16g_gu_half2_dispatches(),
        down_half2: memra_engine::moe_f16g_down_m1_half2_dispatches(),
        wo_a: gpu.dense_wo_a_grouped_dispatches(),
        index_radix: gpu.index_topk_radix_dispatches(),
        small_launches: gpu.small_kernel_launches(),
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
        splitk_gu: after.splitk_gu - before.splitk_gu,
        splitk_down: after.splitk_down - before.splitk_down,
        gu_m1: after.gu_m1 - before.gu_m1,
        gu_half2: after.gu_half2 - before.gu_half2,
        down_half2: after.down_half2 - before.down_half2,
        wo_a: after.wo_a - before.wo_a,
        index_radix: after.index_radix - before.index_radix,
        small_launches: std::array::from_fn(|i| after.small_launches[i] - before.small_launches[i]),
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
    let expected_small = if gpu.small_kernel_diet_enabled() {
        [2, 1]
    } else {
        [6, 2]
    };
    assert_eq!(
        c.small_launches,
        expected_small.map(|n| n * local_steps),
        "HC and Q pack actual enqueues: a diet PASS with the old count is forbidden"
    );
    let splitk = memra_engine::moe_m1_splitk_on();
    let oracle_steps = if splitk { 0 } else { local_steps };
    let splitk_steps = if splitk { local_steps } else { 0 };
    assert_eq!(c.gu_m1, oracle_steps, "GU-M1 actual enqueues");
    assert_eq!(c.gu_half2, oracle_steps, "GU-half2 actual enqueues");
    assert_eq!(c.down_half2, oracle_steps, "down-half2 actual enqueues");
    assert_eq!(c.splitk_gu, splitk_steps, "split-K GU two-pass enqueues");
    assert_eq!(
        c.splitk_down, splitk_steps,
        "split-K down two-pass enqueues"
    );
    println!(
        "MOE_ENGAGEMENT splitk={splitk} gu={} down={}",
        c.splitk_gu, c.splitk_down
    );
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

fn run_once(
    gpu: &Dsv4Gpu,
    prompt: &[u32],
    tokenizer: &Tokenizer,
    repeat: usize,
    sampler_name: &str,
    device: bool,
) -> RunReceipt {
    let mut sampler = device.then(|| gpu.device_sampler().expect("device sampler scratch"));
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
            let token = if let Some(sampler) = &mut sampler {
                gpu.sample_device_logits(&state, sampler, &cfg, &[], None)
            } else {
                dsv4_sample_row(&row, state.pos, &cfg)
            }
            .expect("sample");
            if token == tokenizer.eos_id() {
                eos = true;
                return false;
            }
            generated.push(token);
            if device {
                gpu.decode_step_device_logits(token, &mut state)
                    .expect("TP/EP device sampled decode");
            } else {
                row = gpu
                    .decode_step(token, &mut state)
                    .expect("TP/EP sampled decode");
            }
            if profiled && generated.len() == 64 {
                drain(gpu);
                drop(decode_phase.take());
            }
            true
        },
        || drain(gpu),
    );
    drop(decode_phase);
    let engagements = sampler.as_ref().map_or(0, |s| s.engagements());
    if device {
        assert_eq!(
            engagements as usize,
            generated.len() + usize::from(eos),
            "device sampler must engage on every draw"
        );
        assert!(engagements > 0);
        sampler
            .as_ref()
            .unwrap()
            .check_canary_for_gate()
            .expect("sampler canary");
        row = gpu
            .read_decode_logits_for_gate(&state)
            .expect("final identity row outside timing");
    }
    println!(
        "SAMPLER repeat={repeat} sampler={sampler_name} device_engagements={engagements} logits_d2h_in_decode={}",
        if device { 0 } else { generated.len() }
    );
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
        "MEASURE {{\"repeat\":{repeat},\"sampler_order\":\"{sampler_name}\",\"device_sampler_engagements\":{engagements},\"prompt_tokens\":{PROMPT_TOKENS},\"requested_output_tokens\":{OUTPUT_TOKENS},\"generated_tokens\":{},\"forward_calls\":{},\"eos\":{eos},\"state_alloc_ns\":{},\"prime_wall_ns\":{},\"decode_wall_ns\":{},\"timing_scope\":\"sample_plus_forward_envelope\",\"sampling_in_timing\":true,\"prime_tok_s\":{prime_tok_s:.6},\"decode_tok_s\":{decode_tok_s:.6},\"headline_decode_tok_s\":{headline_decode_tok_s},\"eligible\":{eligible},\"looped\":{is_looped},\"state_pos\":{},\"generated_sha256\":\"{generated_sha256}\",\"final_logits_sha256\":\"{final_logits_sha256}\",\"final_cache_digest\":[{},{}],\"final_hidden_digest\":[{},{}],\"attention_tp\":{attention_mode},\"attention_join_sha256\":{attention_join_json},\"attention_rank_calls\":[{},{}],\"attention_ar_calls\":{},\"ar_refusals\":[{},{}],\"rank_layer_calls\":[{},{}],\"ep_calls\":{},\"ar_dispatches\":{},\"gu_m1_calls\":{},\"gu_half2_calls\":{},\"down_half2_calls\":{},\"wo_a_calls\":{},\"index_radix_calls\":{},\"speculative\":false,\"pp_timing\":false,\"cache_hash_in_timing\":false,\"hidden_hash_in_timing\":false}}",
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

fn select_legacy_dense_control() {
    // This historical instrument owns its frozen dense control program. The
    // composition/default gate measures the ON program explicitly.
    memra_engine::dsv4_gpu::set_dense_exact_tail_for_gate(false).expect("dense control selection");
}

fn select_dense_policy(profile: bool) {
    if profile {
        memra_engine::dsv4_gpu::restore_dense_exact_tail_default_for_gate();
    } else {
        select_legacy_dense_control();
    }
}

fn select_expert_policy(profile: bool, host_splitk: bool) {
    // Profile inherits the environment default, including which split-K entry
    // symbols it captures. Historical scored controls stay OFF and pin the base
    // graph split-K entries so their captured class does not move with the door.
    if !profile {
        memra_engine::set_moe_m1_graph_splitk_for_gate(false);
        memra_engine::set_moe_m1_splitk_fast_for_gate(false);
    }
    memra_engine::set_moe_m1_splitk_for_gate(host_splitk);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let replay_profile = args
        .get(3)
        .is_some_and(|a| a == "--full-token-replay-profile");
    if !replay_profile {
        // Preserve historical OFF controls; profile keeps real unset/0 policy.
        unsafe {
            std::env::set_var("MEMRA_DSV4_HC_DOT_SPLIT", "0");
            std::env::set_var("MEMRA_DSV4_DENSE_FAST", "0");
            std::env::set_var("MEMRA_DSV4_NORM_FUSE", "0");
            std::env::set_var("MEMRA_DSV4_NORM_FUSE2", "0");
            std::env::set_var("MEMRA_DSV4_NORM2_WIDE", "0");
        }
    }
    select_dense_policy(replay_profile);
    if args.get(1).is_some_and(|a| a == "--sampler-component") {
        sampler_component();
        return;
    }
    let cadence_reverse = args
        .get(3)
        .is_some_and(|a| a == "--full-token-replay-cadence-baab");
    let replay_cadence = args
        .get(3)
        .is_some_and(|a| a == "--full-token-replay-cadence")
        || cadence_reverse;
    let replay_reverse = args.get(3).is_some_and(|a| a == "--full-token-replay-baab");
    let full_replay = args.get(3).is_some_and(|a| a == "--full-token-replay")
        || replay_reverse
        || replay_profile
        || replay_cadence;
    let sampler_abba = args.get(3).is_some_and(|a| a == "--sampler-abba");
    assert!(
        args.len() == 3 || args.len() == 4,
        "usage: dsv4_tp_ep_sampled_perf_gate <model-dir> <real-source.txt> [--moe-m1-splitk|--moe-m1-splitk-abba|--sampler-abba|--small-kernel-components|--small-kernel-abba|--full-token-replay|--full-token-replay-baab|--full-token-replay-profile|--full-token-replay-cadence|--full-token-replay-cadence-baab]"
    );
    let components = args
        .get(3)
        .is_some_and(|v| v == "--small-kernel-components");
    let small_abba = args.get(3).is_some_and(|v| v == "--small-kernel-abba");
    let splitk = args.get(3).is_some_and(|a| a == "--moe-m1-splitk");
    let splitk_abba = args.get(3).is_some_and(|a| a == "--moe-m1-splitk-abba");
    assert!(
        args.len() == 3
            || splitk
            || splitk_abba
            || sampler_abba
            || components
            || small_abba
            || full_replay,
        "unknown gate arm"
    );
    select_expert_policy(replay_profile, splitk);
    let attention_mode = match std::env::var("MEMRA_DSV4_ATTENTION_TP_GATE").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("0") => false,
        Ok("1") => true,
        _ => panic!("MEMRA_DSV4_ATTENTION_TP_GATE requires 0 or 1"),
    };
    let device = dsv4_sampler().expect("sampler door") == Dsv4Sampler::Device;
    let sampler = dsv4_sampler_order().expect("explicit sampler configuration");
    let profiled = dsv4_prof_on();
    let host_sampler_name = match sampler {
        Dsv4SamplerOrder::Comparison => "comparison",
        Dsv4SamplerOrder::Radix => "radix",
    };
    let sampler_name = if sampler_abba {
        "host-radix/device"
    } else if device {
        "device"
    } else {
        host_sampler_name
    };
    let repeats = if full_replay {
        20
    } else if sampler_abba || small_abba {
        40
    } else if attention_mode {
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
    let mut gpu = Dsv4Gpu::load(
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

    if full_replay {
        assert!(
            attention_mode && device && (profiled == replay_profile) && !splitk,
            "replay requires TP2/device sampler/split-K OFF and profiling only in the profile-only arm"
        );
        gpu.set_small_kernel_diet_for_gate(true)
            .expect("replay diet");
        if replay_cadence {
            full_token_replay::cadence(&gpu, &prompt[..PROMPT_TOKENS], &tokenizer, cadence_reverse);
        } else if replay_profile {
            full_token_replay::profile(&gpu, &prompt[..PROMPT_TOKENS], &tokenizer);
        } else {
            full_token_replay::run(&gpu, &prompt[..PROMPT_TOKENS], &tokenizer, replay_reverse);
        }
        return;
    }

    if sampler_abba {
        assert!(
            !profiled && attention_mode,
            "ABBA requires unprofiled attention TP2"
        );
        assert_eq!(
            sampler,
            Dsv4SamplerOrder::Radix,
            "CPU radix oracle required"
        );
    }
    if components {
        assert!(
            attention_mode && gpu.chains_f32,
            "component gate requires attention TP2 f32x"
        );
        gpu.enable_small_kernel_components_for_gate();
        let mut state = gpu
            .alloc_decode_state_for_transient(8, 1)
            .expect("component state");
        gpu.prefill_with_cache_chunked(&prompt[..1], &mut state, 1)
            .expect("live checkpoint components");
        drain(&gpu);
        assert!(
            gpu.small_kernel_components_complete_for_gate(),
            "both components must run on both ranks"
        );
        println!(
            "PASS small-kernel live-checkpoint components; both ranks; bitwise; 32 ABBA repeats each"
        );
        return;
    }
    if small_abba {
        assert!(
            attention_mode && !profiled && sampler_name == "radix",
            "diet ABBA requires unprofiled attention TP2 and radix"
        );
    }

    let arms: &[bool] = if splitk_abba {
        &[false, true, true, false]
    } else {
        &[splitk]
    };
    for (arm_index, &armed) in arms.iter().enumerate() {
        gpu.set_grouped_m1_splitk_for_gate(armed);
        println!("ABBA_ARM index={arm_index} splitk={armed} fresh_request_state=true");
        println!(
            "MOE_PROGRAM splitk={armed} component=false numeric_class={}",
            if armed {
                memra_engine::MOE_M1_SPLITK_NUMERIC_CLASS
            } else {
                "existing_m1_f16_mma"
            }
        );
        let receipts: Vec<_> = (0..repeats)
            .map(|repeat| {
                let arm = if sampler_abba {
                    matches!(repeat % 4, 1 | 2)
                } else {
                    device
                };
                if small_abba {
                    gpu.set_small_kernel_diet_for_gate(matches!(repeat % 4, 1 | 2))
                        .expect("ABBA arm");
                }
                let row = run_once(
                    &gpu,
                    &prompt,
                    &tokenizer,
                    repeat,
                    if arm { "device" } else { host_sampler_name },
                    arm,
                );
                let launches: u64 = row.counters_decode.small_launches.iter().sum();
                println!("LAUNCHES repeat={repeat} diet={} targeted_launches={} targeted_launches_per_step_per_rank={:.6} expected_saved_per_layer_per_rank=5 scope=hc_finish_and_q_norm_pack",
                    gpu.small_kernel_diet_enabled(), launches, launches as f64 / (2 * row.forward_calls) as f64);
                row
            })
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
        if sampler_abba {
            assert!(
                receipts.iter().all(|r| r.eligible),
                "all ABBA rows must be eligible"
            );
            let rate = |arm: bool| {
                let rows: Vec<_> = receipts
                    .iter()
                    .filter(|r| matches!(r.repeat % 4, 1 | 2) == arm)
                    .collect();
                rows.iter().map(|r| r.generated_tokens).sum::<usize>() as f64 * 1e9
                    / rows.iter().map(|r| r.decode_wall.as_nanos()).sum::<u128>() as f64
            };
            let host = rate(false);
            let device = rate(true);
            println!(
                "ABBA cycles=10 rows_per_arm=20 host_tok_s={host:.6} device_tok_s={device:.6} delta_pct={:.6} tokens_logits_cache_hidden_identical=true timing_scope=sample_plus_forward_envelope",
                (device / host - 1.0) * 100.0
            );
        }
        if attention_mode && !profiled && !sampler_abba {
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
            if small_abba {
                let mut wall = [0u128; 2];
                let mut tokens = [0usize; 2];
                for row in &receipts {
                    let arm = usize::from(matches!(row.repeat % 4, 1 | 2));
                    wall[arm] += row.decode_wall.as_nanos();
                    tokens[arm] += row.generated_tokens;
                }
                let rates: [f64; 2] =
                    std::array::from_fn(|i| tokens[i] as f64 * 1e9 / wall[i] as f64);
                println!(
                    "ABBA cycles=10 off_tok_s={:.6} on_tok_s={:.6} delta_pct={:.6} digests_identical=true timing_scope=sample_plus_forward_envelope sampler=radix",
                    rates[0],
                    rates[1],
                    100.0 * (rates[1] / rates[0] - 1.0)
                );
            } else {
                println!(
                    "SUMMARY {{\"repeats\":{repeats},\"eligible_repeats\":{repeats},\"generated_tokens\":{total_tokens},\"decode_wall_ns\":{total_wall_ns},\"sampled_envelope_tok_s\":{pooled_tok_s:.6},\"timing_scope\":\"sample_plus_forward_envelope\",\"sampler_order\":\"{sampler_name}\",\"paired_control\":false,\"speculative\":false}}"
                );
            }
        }
        if attention_mode && profiled {
            assert!(receipts.iter().all(|receipt| !receipt.eligible));
            println!(
                "PASS profile-only sampled attention TP2; repeats={repeats} eligible=0 timing_scope=none sampler={sampler_name}"
            );
        } else if attention_mode {
            println!(
                "PASS sampled attention TP2; repeats={repeats} eligible={repeats} timing_scope=sample_plus_forward_envelope sampler={sampler_name}"
            );
        } else {
            println!(
                "PASS sampled TP/EP internal repeat; eligible_first={} eligible_second={} loop exclusion remains per-row eligibility",
                first.eligible, receipts[1].eligible
            );
        }
    }
    Dsv4Gpu::set_attention_tp_for_gate(false);
    Dsv4Gpu::set_tp_ep_topology_for_gate(false);
}

#[cfg(test)]
mod default_policy_tests {
    #[test]
    fn profile_inherits_graph_splitk_policy() {
        const CHILD: &str = "MEMRA_TEST_DSV4_DEFAULT_CHILD";
        if let Ok(expected) = std::env::var(CHILD) {
            super::select_expert_policy(true, false);
            assert_eq!(memra_engine::moe_m1_graph_splitk_on(), expected == "1");
            super::select_expert_policy(false, false);
            assert!(!memra_engine::moe_m1_graph_splitk_on());
            return;
        }
        for value in [None, Some("0"), Some("graph")] {
            let mut child = std::process::Command::new(std::env::current_exe().unwrap());
            child.args([
                "--exact",
                "default_policy_tests::profile_inherits_graph_splitk_policy",
            ]);
            child.env(CHILD, if value == Some("0") { "0" } else { "1" });
            if let Some(value) = value {
                child.env("MEMRA_DSV4_MOE_M1_SPLITK", value);
            } else {
                child.env_remove("MEMRA_DSV4_MOE_M1_SPLITK");
            }
            assert!(child.status().unwrap().success());
        }
    }

    #[test]
    fn environment_defaults_and_explicit_gate_rollback() {
        use memra_engine::dsv4_gpu::{
            dense_exact_tail_enabled_for_gate, dsv4_replay_cadence_default,
            set_dense_exact_tail_for_gate,
        };
        const CHILD: &str = "MEMRA_TEST_DSV4_DEFAULT_CHILD";
        if let Ok(expected) = std::env::var(CHILD) {
            let (cadence, dense) = expected.split_once(':').unwrap();
            assert_eq!(dsv4_replay_cadence_default(), cadence == "1");
            assert_eq!(dense_exact_tail_enabled_for_gate(), dense == "1");
            // The real profile initialization restores the environment policy,
            // even after an explicit override. Other CLI modes stay frozen OFF.
            set_dense_exact_tail_for_gate(dense != "1").unwrap();
            super::select_dense_policy(true);
            assert_eq!(dense_exact_tail_enabled_for_gate(), dense == "1");
            super::select_dense_policy(false);
            assert!(!dense_exact_tail_enabled_for_gate());
            // Exercise the real legacy gate initialization with an ON override.
            set_dense_exact_tail_for_gate(true).unwrap();
            super::select_legacy_dense_control();
            assert!(!dense_exact_tail_enabled_for_gate());
            assert_eq!(
                memra_engine::dsv4_gpu::restore_dense_exact_tail_default_for_gate(),
                dense == "1"
            );
            assert_eq!(dense_exact_tail_enabled_for_gate(), dense == "1");
            // The override is thread-local; a new host thread sees its env default.
            assert_eq!(
                std::thread::spawn(dense_exact_tail_enabled_for_gate)
                    .join()
                    .unwrap(),
                dense == "1"
            );
            return;
        }
        for cadence in [None, Some("0"), Some("1")] {
            for dense in [None, Some("0"), Some("1")] {
                let mut child = std::process::Command::new(std::env::current_exe().unwrap());
                child.args([
                    "--exact",
                    "default_policy_tests::environment_defaults_and_explicit_gate_rollback",
                    "--nocapture",
                ]);
                for (name, value) in [
                    ("MEMRA_DSV4_REPLAY_CADENCE", cadence),
                    ("MEMRA_DSV4_DENSE_EXACT_TAIL", dense),
                ] {
                    if let Some(value) = value {
                        child.env(name, value);
                    } else {
                        child.env_remove(name);
                    }
                }
                child.env(
                    CHILD,
                    format!(
                        "{}:{}",
                        u8::from(cadence != Some("0")),
                        u8::from(dense != Some("0"))
                    ),
                );
                let out = child.output().unwrap();
                assert!(
                    out.status.success(),
                    "cadence={cadence:?} dense={dense:?}: {}",
                    String::from_utf8_lossy(&out.stderr)
                );
            }
        }
    }
}

/// Invert the position-keyed SplitMix64 map to place a draw on a chosen
/// 53-bit uniform value. This constructs boundaries, rather than hoping a seed
/// happens to land near one. The production RNG is unchanged.
fn sampler_boundary_seed(numerator: u64, pos: usize) -> u64 {
    fn undo_xor(y: u64, shift: u32) -> u64 {
        let mut x = y;
        for _ in 0..64u32.div_ceil(shift) {
            x = y ^ (x >> shift);
        }
        x
    }
    fn inverse_odd(a: u64) -> u64 {
        let mut x = 1u64;
        for _ in 0..6 {
            x = x.wrapping_mul(2u64.wrapping_sub(a.wrapping_mul(x)));
        }
        x
    }
    assert!(numerator < (1u64 << 53));
    let z = undo_xor(numerator << 11, 31).wrapping_mul(inverse_odd(0x94d049bb133111eb));
    let z = undo_xor(z, 27).wrapping_mul(inverse_odd(0xbf58476d1ce4e5b9));
    let input = undo_xor(z, 30).wrapping_sub(0x9e3779b97f4a7c15);
    let seed = input ^ (pos as u64).wrapping_mul(0xa24baed4963ee407);
    assert_eq!(
        memra_engine::dsv4_gpu::dsv4_pos_uniform(seed, pos),
        numerator as f64 / (1u64 << 53) as f64,
        "constructed position-keyed draw"
    );
    seed
}

fn sampler_boundary_case(
    n: usize,
    ordinal: usize,
    index: usize,
    pos: usize,
) -> (Vec<f32>, Dsv4SampleCfg, &'static str) {
    let k = 1usize << (3 + (index / 4 + ordinal) % 7);
    let cut = 2 + (index * 7 + ordinal * 11) % (k - 3);
    let start = ordinal * (n / 2) + index * 512;
    let cdf_case = index % 4 == 3;
    let nonuniform = !cdf_case && index & 4 != 0;
    let mut row = vec![-64.0; n];
    for (i, value) in row[start..start + k].iter_mut().enumerate() {
        *value = if nonuniform {
            -(i as f32) / k as f32
        } else {
            0.0
        };
    }
    let mut cfg = Dsv4SampleCfg {
        temperature: 0.75,
        top_p: 1.0,
        top_k: k,
        seed: 20260907 + pos as u64,
    };
    if cdf_case {
        // Exact dyadic probabilities: draw below, at, and above a CDF edge,
        // separated by one RNG quantum (2^-53). Strict u < acc is exercised.
        let numerator = (cut as u64) * ((1u64 << 53) / k as u64) - 1 + ((index / 4) % 3) as u64;
        cfg.seed = sampler_boundary_seed(numerator, pos);
        (row, cfg, "cdf")
    } else {
        // Mirror the oracle's normalization to construct a cumulative mass.
        // top_p is f32: nearest and its two neighbours are within two f32
        // ulps of that f64 mass, and all retain multiple tokens at full vocab.
        let mut probs: Vec<f64> = row[start..start + k]
            .iter()
            .map(|&v| ((v as f64) / cfg.temperature as f64).exp())
            .collect();
        let z: f64 = probs.iter().sum();
        for p in &mut probs {
            *p /= z;
        }
        let boundary: f64 = probs[..cut].iter().sum();
        let nearest = boundary as f32;
        cfg.top_p = match index % 4 {
            0 => nearest.next_down(),
            1 => nearest,
            _ => nearest.next_up(),
        };
        let ulp = (nearest.next_up() as f64 - nearest as f64)
            .max(nearest as f64 - nearest.next_down() as f64);
        assert!((cfg.top_p as f64 - boundary).abs() <= 2.0 * ulp);
        assert!(
            cfg.top_p as f64 > probs[0],
            "nucleus must retain more than one token"
        );
        (
            row,
            cfg,
            if nonuniform {
                "nucleus-exp"
            } else {
                "nucleus-dyadic"
            },
        )
    }
}

#[cfg(test)]
mod sampler_boundary_tests {
    #[test]
    fn constructed_draws_cover_both_sides_and_equality() {
        for pos in [256, 575, 831, 895] {
            for numerator in [(1u64 << 51) - 1, 1u64 << 51, (1u64 << 51) + 1] {
                super::sampler_boundary_seed(numerator, pos);
            }
        }
    }

    #[test]
    fn boundary_cases_have_distinct_gpu_inputs_and_valid_multi_token_nuclei() {
        for index in 0..64 {
            let (row0, cfg0, _) = super::sampler_boundary_case(129280, 0, index, 512 + index);
            let (row1, cfg1, _) = super::sampler_boundary_case(129280, 1, index, 832 + index);
            assert_ne!(row0, row1);
            for cfg in [cfg0, cfg1] {
                assert!(cfg.top_p > 0.0 && cfg.top_p <= 1.0);
                assert!(cfg.top_k > 1);
            }
        }
    }
}

/// Deterministic full-vocabulary tape, regenerated from row and token IDs.
fn sampler_component() {
    use memra_engine::dsv4_gpu::{Dsv4PenaltyCfg, dsv4_penalize_row, dsv4_sample_row_ordered};
    let n = 129280usize;
    let mut total = 0usize;
    for ordinal in 0..2 {
        let ctx = cudarc::driver::CudaContext::new(ordinal).expect("component CUDA context");
        let stream = ctx.default_stream();
        let mut sampler = Dsv4DeviceSampler::new(stream, n).expect("component scratch");
        for r in 0..320usize {
            let case_id = ordinal * 320 + r;
            let pos = 256 + case_id;
            let row: Vec<f32> = (0..n)
                .map(|i| {
                    let x = (i as u64).wrapping_mul(0x9e3779b97f4a7c15)
                        ^ (case_id as u64).wrapping_mul(0xbf58476d1ce4e5b9);
                    match r % 8 {
                        0 => 0.0,
                        1 => {
                            if i % 2 == 0 {
                                -0.0
                            } else {
                                0.0
                            }
                        }
                        2 => (i % 7) as f32,
                        3 => {
                            if i == r {
                                100.0
                            } else {
                                -100.0
                            }
                        }
                        4 => f32::from_bits(1 + (i % 1024) as u32),
                        _ => ((x ^ (x >> 29)) % 32768) as f32 / 1024.0 - 16.0,
                    }
                })
                .collect();
            assert!(row.iter().all(|x| x.is_finite()));
            let cfg = Dsv4SampleCfg {
                temperature: [1.0, 0.01, 10.0, f32::MIN_POSITIVE][(r / 8) % 4],
                top_p: [1.0, 0.9, 1e-7, f32::MIN_POSITIVE][(r / 32) % 4],
                top_k: [0, 1, 37, n + 1][(r / 64) % 4],
                seed: 20260907 + case_id as u64,
            };
            let (row, cfg, boundary) = if r >= 256 {
                sampler_boundary_case(n, ordinal, r - 256, pos)
            } else {
                (row, cfg, "none")
            };
            let penalty = Dsv4PenaltyCfg {
                last_n: 17,
                repeat: 1.1,
                freq: 0.2,
                present: -0.1,
            };
            let window = [0, 1, 1, 3, 3, 3, r as u32, n as u32 + 9];
            let pc = (r < 256 && r % 3 == 0).then_some(&penalty);
            let mut oracle = row.clone();
            if let Some(pc) = pc {
                dsv4_penalize_row(&mut oracle, &window, pc);
            }
            let host = dsv4_sample_row_ordered(&oracle, pos, &cfg, Dsv4SamplerOrder::Radix)
                .expect("host radix");
            let device = sampler
                .sample_host_row(&row, pos, &cfg, &window, pc)
                .expect("device");
            sampler.check_canary_for_gate().expect("component canary");
            println!(
                "COMPONENT gpu={ordinal} row={r} case_id={case_id} boundary={boundary} host={host} device={device} identical={} finite=true canary=true",
                host == device
            );
            assert_eq!(
                host, device,
                "component token identity GPU {ordinal} row {r}"
            );
            total += 1;
        }
        assert_eq!(sampler.engagements(), 320, "component engagement");
    }
    println!(
        "PASS component rows_per_gpu=320 rows={total} identical_tokens=true finite=true canaries=true numeric_class={}",
        memra_engine::dsv4_sampler::NUMERIC_CLASS
    );
}
