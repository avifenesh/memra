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

/// (logits bits sha256, cache digest, hidden digest) of a state's last step.
type Identity = (String, [u64; 2], [u64; 2]);

fn identity(gpu: &Dsv4Gpu, s: &DecodeState) -> Identity {
    let mut hash = Sha256::new();
    for v in gpu.read_decode_logits_for_gate(s).expect("logits") {
        hash.update(v.to_bits().to_le_bytes());
    }
    (
        format!("{:x}", hash.finalize()),
        gpu.tp_ep_cache_digest_for_gate(s).expect("cache digest"),
        gpu.tp_ep_hidden_digest_for_gate(s).expect("hidden digest"),
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
    gpu.set_grouped_route_validation_for_gate(false);
    gpu.set_grouped_mirror_validation_for_gate(false);
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
    let cfg = Dsv4SampleCfg {
        temperature: 1.,
        top_p: 1.,
        top_k: 0,
        seed: 20260924,
    };
    println!(
        "PROTOCOL {{\"prefix\":{PREFIX},\"steps\":{steps},\"capacity\":{capacity},\"compare\":\"token, logits bits, TP/EP cache and hidden digests per step\",\"seed\":{},\"source_sha256\":\"{}\"}}",
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
    let mut first = gpu
        .sample_device_logits(&prefix, &mut sampler, &cfg, &[], None)
        .expect("sample");
    while prefix.pos < PREFIX {
        gpu.decode_step_device_logits(first, &mut prefix)
            .expect("prefix step");
        first = gpu
            .sample_device_logits(&prefix, &mut sampler, &cfg, &[], None)
            .expect("sample");
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
        gpu.arm_full_token_replay_for_gate(&mut replay, cfg)
            .expect("arm replay");
    }
    let before = gpu
        .full_token_replay_variant_counts_for_gate(&replay)
        .expect("counts");

    // The eager arm steps and samples through the unarmed TP/EP program; the draw is keyed on
    // (seed, position), so equal logits bits give the replay's token.
    let mut eager_sampler = gpu.device_sampler().expect("eager sampler");
    let (mut te, mut tr) = (first, first);
    let mut tokens = Vec::with_capacity(steps);
    let mut first_bad = None;
    for step in 0..steps {
        let pos = eager.pos;
        tokens.push(te);
        gpu.decode_step_device_logits(te, &mut eager)
            .expect("eager step");
        te = gpu
            .sample_device_logits(&eager, &mut eager_sampler, &cfg, &[], None)
            .expect("eager sample");
        tr = gpu
            .decode_sample_full_token_for_gate(tr, &mut replay)
            .expect("replay step");
        let (ie, ir) = (identity(&gpu, &eager), identity(&gpu, &replay));
        if te != tr || ie != ir {
            first_bad = Some((step, pos, te, tr, ie, ir));
            break;
        }
    }
    let after = gpu
        .full_token_replay_variant_counts_for_gate(&replay)
        .expect("counts");
    let delta: Vec<[u64; 4]> = (0..2)
        .map(|r| {
            let mut d = [0u64; 4];
            for v in 0..4 {
                d[v] = after[r][v] - before[r][v];
            }
            d
        })
        .collect();
    let captures = gpu
        .full_token_replay_captures_for_gate(&replay)
        .expect("captures");
    println!(
        "REPLAY_VARIANTS rank0={:?} rank1={:?} (ordinary, commit, c4, c4+c128) captures={captures:?}",
        delta[0], delta[1]
    );
    let mut hash = Sha256::new();
    for t in &tokens {
        hash.update(t.to_le_bytes());
    }
    let tokens_sha = format!("{:x}", hash.finalize());
    if let Some((step, pos, te, tr, ie, ir)) = first_bad {
        println!(
            "FIRST DIVERGENCE step={step} pos={pos} eager=(tok {te}, {ie:?}) replay=(tok {tr}, {ir:?})"
        );
        println!("FAILED: replay diverged from the eager full-token program");
        std::process::exit(1);
    }
    let replayed: Vec<u64> = delta.iter().map(|d| d[0] + d[2] + d[3]).collect();
    if replayed.iter().any(|&n| n != steps as u64) {
        println!("FAILED: {replayed:?} replayed steps per rank, expected {steps} each");
        std::process::exit(1);
    }
    println!(
        "PASS: {steps} replayed steps from position {PREFIX} to {} bit-identical to eager (token, logits, cache and hidden digests); tokens_sha256={tokens_sha}",
        PREFIX + steps
    );

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
                gpu.decode_sample_full_token_for_gate(tok, state)
                    .expect("replay step")
            } else {
                gpu.decode_step_device_logits(tok, state)
                    .expect("eager step");
                gpu.sample_device_logits(state, &mut sampler, &cfg, &[], None)
                    .expect("eager sample")
            };
        }
        1e3 * t0.elapsed().as_secs_f64() / steps as f64
    };
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
