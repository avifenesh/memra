//! TP/EP position-split C4 store identity gate (memra #710).
//!
//! Under TP/EP every trunk layer's caches live on both ranks. A split state stores each C4
//! layer's compressed blocks by parity, even on rank 0 and odd on rank 1, and gathers the rows
//! a query selected from both. The claim is that the numeric program is unchanged.
//!
//! On one loaded TP/EP program:
//! - A replicated state (the reference) and a split state prefill the same prompt in chunks,
//!   then decode greedy steps eagerly. The prefill logits and every step's full logits must
//!   match bit for bit.
//! - A second split state primed the same way decodes on the full-token replay graphs and must
//!   give the same bits.
//! - Both eager states are parked to host and restored, and the restored pair's continuation
//!   must match too.
//!
//! The prompt is long enough that the indexer selects from more blocks than its top-k, so
//! remote rows, recent-ring rows and local rows all reach attention.
//!
//! Usage: `dsv4_kv_split_gate <model-dir> <source-tape> [prompt-tokens] [steps]` (defaults
//! 3000, 300). Rig law: under the box GPU lock, one pair, no other tenant.
use memra_engine::dsv4_gpu::{DecodeState, Dsv4Gpu, Dsv4SampleCfg, set_c4_split_for_gate};
use memra_engine::dsv4_source_tape::SourceTape;
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use std::path::Path;

const CHUNK: usize = 512;

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

fn bits(row: &[f32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for v in row {
        for byte in v.to_bits().to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn argmax(row: &[f32]) -> u32 {
    let mut best = 0usize;
    for i in 1..row.len() {
        if row[i] > row[best] {
            best = i;
        }
    }
    best as u32
}

fn prime(gpu: &Dsv4Gpu, prompt: &[u32], capacity: usize, split: bool) -> (DecodeState, u32, u64) {
    set_c4_split_for_gate(Some(split));
    let mut state = gpu
        .alloc_decode_state_for_transient(capacity, CHUNK)
        .expect("decode state");
    set_c4_split_for_gate(None);
    let logits = gpu
        .prefill_with_cache_chunked(prompt, &mut state, CHUNK)
        .expect("chunked prefill");
    (state, argmax(&logits), bits(&logits))
}

/// Greedy eager steps: (token, logits hash) per step.
fn eager(gpu: &Dsv4Gpu, state: &mut DecodeState, mut tok: u32, steps: usize) -> Vec<(u32, u64)> {
    (0..steps)
        .map(|_| {
            let logits = gpu.decode_step(tok, state).expect("eager step");
            tok = argmax(&logits);
            (tok, bits(&logits))
        })
        .collect()
}

fn first_divergence(a: &[(u32, u64)], b: &[(u32, u64)]) -> Option<usize> {
    (0..a.len().min(b.len())).find(|&i| a[i] != b[i])
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(
        (3..=5).contains(&args.len()),
        "usage: dsv4_kv_split_gate <model-dir> <source-tape> [prompt-tokens] [steps]"
    );
    let prompt_tokens: usize = args
        .get(3)
        .map_or(3000, |v| v.parse().expect("prompt tokens"));
    let steps: usize = args.get(4).map_or(300, |v| v.parse().expect("steps"));
    for (key, value) in PINS {
        match std::env::var(key) {
            Ok(v) => assert_eq!(v, value, "{key} must be {value}"),
            Err(_) => unsafe { std::env::set_var(key, value) },
        }
    }
    let dir = Path::new(&args[1]);
    let tape = SourceTape::read(&args[2]).expect("source tape");
    let tokenizer = Tokenizer::from_hf_dir(dir).expect("tokenizer");
    let base = tape.prompt(&tokenizer, "Review this inference engine source:\n\n", 256);
    let mut prompt = Vec::with_capacity(prompt_tokens);
    while prompt.len() < prompt_tokens {
        let take = (prompt_tokens - prompt.len()).min(base.len());
        prompt.extend_from_slice(&base[..take]);
    }
    let capacity = (prompt_tokens + 2 * steps + 256).next_multiple_of(512);

    Dsv4Gpu::set_tp_ep_topology_for_gate(true);
    Dsv4Gpu::set_attention_tp_for_gate(true);
    let gpu =
        Dsv4Gpu::load(dir, &[0, 1], ActQuantVariant::RefFp8Round, capacity + 64).expect("load");
    assert!(gpu.topology().is_tp_ep() && gpu.attention_tp_geometry().is_some());
    println!(
        "PROTOCOL {{\"prompt_tokens\":{prompt_tokens},\"chunk\":{CHUNK},\"steps\":{steps},\"capacity\":{capacity},\"compare\":\"prefill and every step's full logits bits, replicated against split\",\"source_sha256\":\"{}\"}}",
        tape.sha256
    );

    // The session byte plans at the served context, per rank (planning only, no allocation).
    let plan = |split: bool, cap: usize| {
        set_c4_split_for_gate(Some(split));
        let (dev, _) = gpu
            .plan_session_cache_bytes(cap, CHUNK, false)
            .expect("session plan");
        set_c4_split_for_gate(None);
        dev
    };
    for cap in [65536usize, 262144, 1048576] {
        if cap <= gpu.max_seq {
            let (r, sp) = (plan(false, cap), plan(true, cap));
            println!(
                "PLAN capacity={cap} replicated={r:?} split={sp:?} bytes/token/rank replicated={:.0} split={:.0}",
                r[0] as f64 / cap as f64,
                sp[0] as f64 / cap as f64
            );
        }
    }
    let (mut rep, t_rep, pre_rep) = prime(&gpu, &prompt, capacity, false);
    let (mut spl, t_spl, pre_spl) = prime(&gpu, &prompt, capacity, true);
    let bytes = |s: &DecodeState| s.cache_bytes.iter().sum::<u64>();
    println!(
        "CACHE BYTES replicated={} split={} ({:.1}% of replicated)",
        bytes(&rep),
        bytes(&spl),
        100.0 * bytes(&spl) as f64 / bytes(&rep) as f64
    );
    if (t_rep, pre_rep) != (t_spl, pre_spl) {
        println!("FAILED: prefill logits differ (replicated {pre_rep:016x}, split {pre_spl:016x})");
        std::process::exit(1);
    }
    println!(
        "PASS: chunked prefill of {prompt_tokens} tokens bit-identical, split against replicated"
    );

    let a = eager(&gpu, &mut rep, t_rep, steps);
    let b = eager(&gpu, &mut spl, t_spl, steps);
    if let Some(i) = first_divergence(&a, &b) {
        println!(
            "FAILED: eager step {i} replicated {:?} split {:?}",
            a[i], b[i]
        );
        std::process::exit(1);
    }
    println!("PASS: {steps} eager steps bit-identical, split against replicated");

    // Full-token replay on a split state primed the same way.
    let (mut rp, t_rp, _) = prime(&gpu, &prompt, capacity, true);
    let greedy = Dsv4SampleCfg {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        seed: 0,
    };
    unsafe { gpu.arm_full_token_replay(&mut rp, greedy) }.expect("arm replay on the split state");
    let mut tok = t_rp;
    let c: Vec<(u32, u64)> = (0..steps)
        .map(|_| {
            tok = gpu
                .decode_sample_full_token(tok, &mut rp)
                .expect("replayed split step");
            let logits = gpu.read_decode_logits_for_gate(&rp).expect("replay logits");
            (tok, bits(&logits))
        })
        .collect();
    if let Some(i) = first_divergence(&a, &c) {
        println!(
            "FAILED: replay step {i} replicated {:?} split replay {:?}",
            a[i], c[i]
        );
        std::process::exit(1);
    }
    println!(
        "PASS: {steps} full-token replay steps on the split state bit-identical to replicated eager"
    );

    // Park and restore both eager states, then continue.
    let park_rep = gpu.snapshot_decode_state(&rep).expect("park replicated");
    let park_spl = gpu.snapshot_decode_state(&spl).expect("park split");
    set_c4_split_for_gate(Some(false));
    let mut rep2 = gpu
        .restore_decode_state(&park_rep)
        .expect("restore replicated");
    set_c4_split_for_gate(Some(true));
    let mut spl2 = gpu.restore_decode_state(&park_spl).expect("restore split");
    set_c4_split_for_gate(None);
    let tail = 50.min(capacity - rep2.pos - 1);
    let d = eager(&gpu, &mut rep2, a[steps - 1].0, tail);
    let e = eager(&gpu, &mut spl2, b[steps - 1].0, tail);
    if let Some(i) = first_divergence(&d, &e) {
        println!(
            "FAILED: restored step {i} replicated {:?} split {:?}",
            d[i], e[i]
        );
        std::process::exit(1);
    }
    println!("PASS: park, restore and {tail} more steps bit-identical, split against replicated");
    println!("PASS: position-split C4 store bit-identical to the replicated program");
}
