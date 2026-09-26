//! TP/EP full-token replay past the old 512-position arming cap (memra #710).
//!
//! One eager prefix to position 400 is restored into two states. State E steps and samples the
//! continuation through the unarmed TP/EP program (`decode_step_device_logits` and
//! `sample_device_logits`); state R decodes through the armed per-rank replay graphs, which
//! step and sample in one call. Every step must give the same sampled token, the same
//! logits bits and the same TP/EP cache and hidden digests, and every R step must be a replay
//! (the per-variant counters advance by the step count). The run covers positions 400 to
//! 400 + steps, so it crosses 512 and the C4 and C128 emission cadences.
//!
//! Usage: `dsv4_tp_replay_long_gate <model-dir> <source-tape> [steps]` (default 304).
//! `DSV4_REPLAY_GATE_MOE=stream|sktail` picks the expert program (default: the served stream
//! visitor); the two must give the same tokens and digests.
//!
//! Past its replay limit (`min(capacity, 16384)`, or `DSV4_REPLAY_GATE_LIMIT`) state R drops
//! its graphs and continues on the eager step, exactly as the served route does, and every
//! later step must still match state E. With the limit inside the run the gate checks the
//! replay-to-eager handoff; the timing arm then does not run.
//!
//! Profile mode (`DSV4_REPLAY_GATE_PROFILE=replay|eager`): after the identity arm, one warm run
//! of that arm, then one bracketed by cuProfilerStart/Stop, in place of the timing arm.
//! Rig law: under the box GPU lock, one pair, no other tenant.
use memra_engine::dsv4_gpu::{DecodeState, Dsv4Gpu, Dsv4SampleCfg};
use memra_engine::dsv4_source_tape::SourceTape;
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::path::Path;

const PREFIX: usize = 400;

/// The pinned plain TP2 / expert-id EP program the full-token replay admits.
const PINS: [(&str, &str); 7] = [
    ("MEMRA_DSV4_DECODE_PATH", "device"),
    ("MEMRA_DSV4_EXPERT_ARM", "native"),
    ("MEMRA_DSV4_DENSE_ARM", "fp8"),
    ("MEMRA_DSV4_EP", "pair"),
    ("MEMRA_DSV4_GROUPED_ROUTE", "device"),
    ("MEMRA_DSV4_VERIFY_TOPK", "device"),
    ("MEMRA_DSV4_DRAFTER", "off"),
];

/// (logits bits sha256, cache digest, hidden digest) of a state's last step. The digests are
/// `None` on steps the digest cadence skips.
type Identity = (String, Option<([u64; 2], [u64; 2])>);

fn identity(gpu: &Dsv4Gpu, s: &DecodeState, digests: bool) -> Identity {
    let mut hash = Sha256::new();
    for v in gpu.read_decode_logits_for_gate(s).expect("logits") {
        hash.update(v.to_bits().to_le_bytes());
    }
    (
        format!("{:x}", hash.finalize()),
        digests.then(|| {
            (
                gpu.tp_ep_cache_digest_for_gate(s).expect("cache digest"),
                gpu.tp_ep_hidden_digest_for_gate(s).expect("hidden digest"),
            )
        }),
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(
        (3..=4).contains(&args.len()),
        "usage: dsv4_tp_replay_long_gate <model-dir> <source-tape> [steps]"
    );
    let steps: usize = args.get(3).map_or(304, |v| v.parse().expect("steps"));
    // DSV4_REPLAY_GATE_CAPACITY: the session capacity replay arms for (default 1024).
    let capacity: usize = std::env::var("DSV4_REPLAY_GATE_CAPACITY")
        .map_or(1024, |v| v.parse().expect("DSV4_REPLAY_GATE_CAPACITY"));
    assert!(
        PREFIX + steps < capacity,
        "steps must stay inside the session capacity"
    );
    // DSV4_REPLAY_GATE_LIMIT: a replay limit below the served one, so the run crosses the
    // replay-to-eager handoff without a 16384-step walk.
    let limit_override: Option<usize> = std::env::var("DSV4_REPLAY_GATE_LIMIT")
        .ok()
        .map(|v| v.parse().expect("DSV4_REPLAY_GATE_LIMIT"));
    let limit = limit_override.unwrap_or(capacity.min(16384));
    assert!(limit > PREFIX, "the replay limit must lie past the prefix");
    // DSV4_REPLAY_GATE_DIGEST_EVERY=N: the cache and hidden digests every N steps and on the 64
    // steps either side of the handoff (default every step). The digests read every layer's
    // caches, so a 16k-step run at a long capacity checks tokens and logits bits every step and
    // the digests on that cadence.
    let digest_every: usize = std::env::var("DSV4_REPLAY_GATE_DIGEST_EVERY")
        .map_or(1, |v| v.parse().expect("DSV4_REPLAY_GATE_DIGEST_EVERY"));
    assert!(digest_every >= 1);
    let replay_steps = steps.min(limit - PREFIX);
    // Process startup, before any model or worker thread exists: pin the admitted program.
    for (key, value) in PINS {
        match std::env::var(key) {
            Ok(v) => assert_eq!(v, value, "{key} must be {value} for full-token replay"),
            Err(_) => unsafe { std::env::set_var(key, value) },
        }
    }
    assert!(
        std::env::var_os("MEMRA_DSV4_REPLAY_CADENCE").is_none(),
        "MEMRA_DSV4_REPLAY_CADENCE must be unset (the default cadence is the served one)"
    );
    let dir = Path::new(&args[1]);
    let tape = SourceTape::read(&args[2]).expect("source tape");
    let tokenizer = Tokenizer::from_hf_dir(dir).expect("tokenizer");
    let prompt = tape.prompt(&tokenizer, "Review this inference engine source:\n\n", 256);

    Dsv4Gpu::set_tp_ep_topology_for_gate(true);
    Dsv4Gpu::set_attention_tp_for_gate(true);
    let gpu =
        Dsv4Gpu::load(dir, &[0, 1], ActQuantVariant::RefFp8Round, capacity + 64).expect("load");
    assert!(gpu.topology().is_tp_ep() && gpu.attention_tp_geometry().is_some());
    // DSV4_REPLAY_GATE_VALIDATION: `on` (default, the served program: route and mirror checks
    // deferred to device fault words) or `off` (the unchecked arm replay was first qualified on).
    let validation = std::env::var("DSV4_REPLAY_GATE_VALIDATION").unwrap_or_else(|_| "on".into());
    match validation.as_str() {
        "on" => {}
        "off" => {
            gpu.set_grouped_route_validation_for_gate(false);
            gpu.set_grouped_mirror_validation_for_gate(false);
        }
        other => panic!("DSV4_REPLAY_GATE_VALIDATION {other:?} must be on or off"),
    }
    println!("VALIDATION {validation}");
    // DSV4_REPLAY_GATE_MOE: `stream` (default) is the served one-token stream visitor;
    // `sktail` pins the gate-only graph split-K set the replay was first qualified on.
    let moe = std::env::var("DSV4_REPLAY_GATE_MOE").unwrap_or_else(|_| "stream".into());
    match moe.as_str() {
        "stream" => {}
        "sktail" => {
            gpu.set_grouped_gu_fuse_for_gate(true);
            gpu.set_grouped_m1_tc_for_gate(true);
            memra_engine::set_moe_f16g_gu_m1_tc_for_gate(true);
            memra_engine::set_moe_f16g_gu_half2_for_gate(true);
            memra_engine::set_moe_f16g_down_m1_half2_for_gate(true);
        }
        other => panic!("DSV4_REPLAY_GATE_MOE {other:?} must be stream or sktail"),
    }
    println!("MOE {moe}");
    // DSV4_REPLAY_GATE_SAMPLING: `default` (temperature 1, the vendor default) or `greedy`
    // (temperature 0: the eager arm runs the device argmax step, replay captures the argmax).
    let sampling = std::env::var("DSV4_REPLAY_GATE_SAMPLING").unwrap_or_else(|_| "default".into());
    let greedy = match sampling.as_str() {
        "default" => false,
        "greedy" => true,
        other => panic!("DSV4_REPLAY_GATE_SAMPLING {other:?} must be default or greedy"),
    };
    println!("SAMPLING {sampling}");
    let cfg = Dsv4SampleCfg {
        temperature: if greedy { 0. } else { 1. },
        top_p: 1.,
        top_k: 0,
        seed: 20260924,
    };
    println!(
        "PROTOCOL {{\"prefix\":{PREFIX},\"steps\":{steps},\"capacity\":{capacity},\"replay_limit\":{limit},\"digest_every\":{digest_every},\"compare\":\"token and logits bits per step, TP/EP cache and hidden digests on the digest cadence and around the handoff\",\"seed\":{},\"source_sha256\":\"{}\"}}",
        cfg.seed, tape.sha256
    );

    // Eager prefix: the prompt, then sampled tokens up to PREFIX.
    let mut prefix = gpu
        .alloc_decode_state_for_transient(capacity, 1)
        .expect("prefix state");
    gpu.prefill_with_cache_chunked(&prompt[..1], &mut prefix, 1)
        .expect("prefill");
    for &token in &prompt[1..256] {
        gpu.decode_step_device_logits(token, &mut prefix)
            .expect("prompt step");
    }
    let mut sampler = gpu.device_sampler().expect("sampler");
    let mut first = if greedy {
        let logits = gpu.read_decode_logits_for_gate(&prefix).expect("logits");
        (0..logits.len()).fold(0, |b, i| if logits[i] > logits[b] { i } else { b }) as u32
    } else {
        gpu.sample_device_logits(&prefix, &mut sampler, &cfg, &[], None)
            .expect("sample")
    };
    while prefix.pos < PREFIX {
        first = if greedy {
            gpu.decode_step_greedy(first, &mut prefix)
                .expect("prefix step")
        } else {
            gpu.decode_step_device_logits(first, &mut prefix)
                .expect("prefix step");
            gpu.sample_device_logits(&prefix, &mut sampler, &cfg, &[], None)
                .expect("sample")
        };
    }

    let mut eager = gpu
        .alloc_decode_state_for_transient(capacity, 1)
        .expect("eager state");
    let mut replay = gpu
        .alloc_decode_state_for_transient(capacity, 1)
        .expect("replay state");
    gpu.restore_full_token_prefix_for_gate(&mut eager, &prefix)
        .expect("restore eager");
    gpu.restore_full_token_prefix_for_gate(&mut replay, &prefix)
        .expect("restore replay");
    unsafe {
        match limit_override {
            Some(limit) => gpu.arm_full_token_replay_limit_for_gate(&mut replay, cfg, limit),
            None => gpu.arm_full_token_replay(&mut replay, cfg),
        }
        .expect("arm replay");
    }
    let before = gpu
        .full_token_replay_variant_counts_for_gate(&replay)
        .expect("counts");

    // The eager arm steps and samples through the unarmed TP/EP program; the draw is keyed on
    // (seed, position), so equal logits bits give the replay's token.
    let mut eager_sampler = gpu.device_sampler().expect("eager sampler");
    let mut handoff_sampler = gpu.device_sampler().expect("handoff sampler");
    let eager_step = |tok: u32, state: &mut DecodeState, sampler: &mut _| -> u32 {
        if greedy {
            gpu.decode_step_greedy(tok, state).expect("eager step")
        } else {
            gpu.decode_step_device_logits(tok, state)
                .expect("eager step");
            gpu.sample_device_logits(state, sampler, &cfg, &[], None)
                .expect("eager sample")
        }
    };
    let (mut te, mut tr) = (first, first);
    let mut tokens = Vec::with_capacity(steps);
    // Every step's logits hash and digests, so two processes (a launch-scheduling door on and
    // off, two binaries) can compare their whole numeric program, not only their tokens.
    let mut program = Sha256::new();
    let mut first_bad = None;
    let mut armed = true;
    let mut after = before;
    let mut captures = None;
    let mut handoff = None;
    for step in 0..steps {
        let pos = eager.pos;
        tokens.push(te);
        te = eager_step(te, &mut eager, &mut eager_sampler);
        // The served route's handoff: once the replay no longer covers the position, read
        // the counters, drop the graphs and step eager.
        if armed && !gpu.full_token_replay_covers(&replay) {
            after = gpu
                .full_token_replay_variant_counts_for_gate(&replay)
                .expect("counts");
            captures = Some(
                gpu.full_token_replay_captures_for_gate(&replay)
                    .expect("captures"),
            );
            gpu.disarm_full_token_replay(&mut replay)
                .expect("disarm replay");
            armed = false;
            handoff = Some(replay.pos);
        }
        tr = if armed {
            gpu.decode_sample_full_token(tr, &mut replay)
                .expect("replay step")
        } else {
            eager_step(tr, &mut replay, &mut handoff_sampler)
        };
        let digests = step % digest_every == 0 || pos.abs_diff(limit) <= 64;
        let (ie, ir) = (
            identity(&gpu, &eager, digests),
            identity(&gpu, &replay, digests),
        );
        if te != tr || ie != ir {
            first_bad = Some((step, pos, te, tr, ie, ir));
            break;
        }
        program.update(format!("{step} {te} {ie:?}\n").as_bytes());
    }
    if armed {
        after = gpu
            .full_token_replay_variant_counts_for_gate(&replay)
            .expect("counts");
        captures = Some(
            gpu.full_token_replay_captures_for_gate(&replay)
                .expect("captures"),
        );
    }
    println!("HANDOFF {handoff:?} (the position replay handed the request to the eager step)");
    let delta: Vec<[u64; 4]> = (0..2)
        .map(|r| {
            let mut d = [0u64; 4];
            for v in 0..4 {
                d[v] = after[r][v] - before[r][v];
            }
            d
        })
        .collect();
    let captures = captures.expect("captures read while armed");
    println!(
        "REPLAY_VARIANTS rank0={:?} rank1={:?} (ordinary, commit, c4, c4+c128) captures={captures:?}",
        delta[0], delta[1]
    );
    let mut hash = Sha256::new();
    for t in &tokens {
        hash.update(t.to_le_bytes());
    }
    let tokens_sha = format!("{:x}", hash.finalize());
    println!(
        "PROGRAM_SHA256 {:x} (every step's token, logits bits hash and digests)",
        program.finalize()
    );
    if let Some((step, pos, te, tr, ie, ir)) = first_bad {
        println!(
            "FIRST DIVERGENCE step={step} pos={pos} eager=(tok {te}, {ie:?}) replay=(tok {tr}, {ir:?})"
        );
        println!("FAILED: replay diverged from the eager full-token program");
        std::process::exit(1);
    }
    let replayed: Vec<u64> = delta.iter().map(|d| d[0] + d[2] + d[3]).collect();
    if replayed.iter().any(|&n| n != replay_steps as u64) {
        println!("FAILED: {replayed:?} replayed steps per rank, expected {replay_steps} each");
        std::process::exit(1);
    }
    if (replay_steps < steps) != handoff.is_some() || handoff.is_some_and(|p| p != limit) {
        println!(
            "FAILED: handoff at {handoff:?}, expected one at {limit} only when the run passes it"
        );
        std::process::exit(1);
    }
    println!(
        "PASS: {steps} steps from position {PREFIX} to {} bit-identical to eager (token and logits every step, cache and hidden digests {}), {replay_steps} of them replayed{}; tokens_sha256={tokens_sha}",
        PREFIX + steps,
        if digest_every == 1 {
            "every step".to_string()
        } else {
            format!("every {digest_every} steps and around the handoff")
        },
        if handoff.is_some() {
            format!(", then eager from the handoff at {limit}")
        } else {
            String::new()
        }
    );
    if handoff.is_some() {
        return;
    }

    // Timing: the same continuation from the same restored prefix, eager and replay in
    // alternating order, wall time per token (each step returns its sampled token to the host).
    let run = |armed: bool, eager: &mut DecodeState, replay: &mut DecodeState| -> f64 {
        let state = if armed { replay } else { eager };
        gpu.restore_full_token_prefix_for_gate(state, &prefix)
            .expect("restore");
        let mut sampler = gpu.device_sampler().expect("sampler");
        let mut tok = first;
        let t0 = std::time::Instant::now();
        for _ in 0..steps {
            tok = if armed {
                gpu.decode_sample_full_token(tok, state)
                    .expect("replay step")
            } else if greedy {
                gpu.decode_step_greedy(tok, state).expect("eager step")
            } else {
                gpu.decode_step_device_logits(tok, state)
                    .expect("eager step");
                gpu.sample_device_logits(state, &mut sampler, &cfg, &[], None)
                    .expect("eager sample")
            };
        }
        1e3 * t0.elapsed().as_secs_f64() / steps as f64
    };
    // Profile mode (`DSV4_REPLAY_GATE_PROFILE=replay|eager`): one run of that arm bracketed
    // by cuProfilerStart/Stop for `nsys --capture-range=cudaProfilerApi`, after the identity arm.
    if let Ok(arm) = std::env::var("DSV4_REPLAY_GATE_PROFILE") {
        let armed = match arm.as_str() {
            "replay" => true,
            "eager" => false,
            other => panic!("DSV4_REPLAY_GATE_PROFILE {other:?} must be replay or eager"),
        };
        run(armed, &mut eager, &mut replay);
        cudarc::driver::profiler_start().expect("cuProfilerStart");
        let ms = run(armed, &mut eager, &mut replay);
        cudarc::driver::profiler_stop().expect("cuProfilerStop");
        println!("PROFILE arm={arm} steps={steps} ms_per_token={ms:.3}");
        return;
    }
    for rep in 0..3 {
        let order = if rep % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        };
        let mut ms = [0f64; 2];
        for armed in order {
            ms[usize::from(armed)] = run(armed, &mut eager, &mut replay);
        }
        println!(
            "TIME rep={rep} eager_ms_per_token={:.3} replay_ms_per_token={:.3} eager_tok_s={:.2} replay_tok_s={:.2} speedup={:.3}",
            ms[0],
            ms[1],
            1e3 / ms[0],
            1e3 / ms[1],
            ms[0] / ms[1]
        );
    }
}
