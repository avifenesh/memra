//! B-row decode identity gate (memra #667 lever 2).
//!
//! Four sessions with distinct prompts of distinct lengths decode `steps` greedy tokens
//! alone, one `decode_step` each (the served one-row matrix program). Each step's full
//! logits row is hashed bit for bit. Then fresh copies of the same sessions decode again
//! through `decode_rows_logits`: session s joins at step `JOIN[s]` and leaves after its
//! `steps` tokens, so the batch width runs 1, 2, 3, 4 and back down. The row order rotates
//! every step, so a session sits at every row index.
//!
//! Correctness: every session's logits bits and tokens at every step equal its solo run.
//! A request must not change numeric program when a peer joins or leaves, or when its row
//! moves.
//!
//! Timing (after the identity arms, decode wall only): the one-row step against B-row steps
//! of 1, 2 and 4 rows, interleaved, reported as ms per step and aggregate tokens per second.
//!
//! Usage: `dsv4_rows_gate <model-dir> <source.txt> [steps] [timing-steps]`.
//! Rig law: under the box GPU lock, served defaults (no MEMRA_DSV4_* overrides).
use memra_engine::dsv4_gpu::{DecodeState, Dsv4Gpu};
use memra_engine::dsv4_source_tape::SourceTape;
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use std::{path::Path, time::Instant};

const CHUNK: usize = 256;
const LENS: [usize; 4] = [256, 197, 311, 150];
const JOIN: [usize; 4] = [0, 3, 7, 12];
const FRAMES: [&str; 4] = [
    "Review this inference engine source:\n\n",
    "Explain what this inference engine source does:\n\n",
    "List the invariants this source keeps:\n\n",
    "Summarize this file for a new contributor:\n\n",
];

fn drain(gpu: &Dsv4Gpu) {
    for stage in &gpu.stages {
        stage.gpu.stream().synchronize().expect("gate drain");
    }
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

/// FNV-1a over the logits' bit patterns: two rows hash equal only if every bit matches
/// (up to a 64-bit collision).
fn bits_hash(row: &[f32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for v in row {
        for byte in v.to_bits().to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

struct Session {
    state: DecodeState,
    next: u32,
}

fn prime(gpu: &Dsv4Gpu, prompt: &[u32], capacity: usize) -> Session {
    let mut state = gpu
        .alloc_decode_state_for_transient(capacity, CHUNK.max(gpu.verify_tmax()))
        .expect("decode state");
    let logits = gpu
        .prefill_with_cache_chunked(prompt, &mut state, CHUNK)
        .expect("prefill");
    Session {
        state,
        next: argmax(&logits),
    }
}

/// (token, logits bit hash) per step, per session.
type Trace = Vec<Vec<(u32, u64)>>;

fn solo(gpu: &Dsv4Gpu, prompts: &[Vec<u32>], capacity: usize, steps: usize) -> Trace {
    prompts
        .iter()
        .map(|p| {
            let mut s = prime(gpu, p, capacity);
            (0..steps)
                .map(|_| {
                    let logits = gpu.decode_step(s.next, &mut s.state).expect("solo step");
                    s.next = argmax(&logits);
                    (s.next, bits_hash(&logits))
                })
                .collect()
        })
        .collect()
}

fn batched(
    gpu: &Dsv4Gpu,
    prompts: &[Vec<u32>],
    capacity: usize,
    steps: usize,
) -> (Trace, Vec<usize>) {
    let mut sessions: Vec<Session> = prompts.iter().map(|p| prime(gpu, p, capacity)).collect();
    let mut rows = gpu
        .alloc_rows_state(sessions.len())
        .expect("B-row workspace");
    let mut trace: Trace = vec![Vec::new(); sessions.len()];
    let mut widths = Vec::new();
    let horizon = JOIN.iter().max().unwrap() + steps;
    for k in 0..horizon {
        let mut active: Vec<usize> = (0..sessions.len())
            .filter(|&s| k >= JOIN[s] && trace[s].len() < steps)
            .collect();
        if active.is_empty() {
            continue;
        }
        let n = active.len();
        active.rotate_left(k % n);
        widths.push(n);
        let toks: Vec<u32> = active.iter().map(|&s| sessions[s].next).collect();
        let mut picked: Vec<Option<&mut Session>> = sessions.iter_mut().map(Some).collect();
        let mut states: Vec<&mut DecodeState> = active
            .iter()
            .map(|&s| &mut picked[s].take().expect("each session once").state)
            .collect();
        let logits = gpu
            .decode_rows_logits(&toks, &mut states, &mut rows)
            .expect("B-row step");
        drop(states);
        for (&s, row) in active.iter().zip(&logits) {
            let tok = argmax(row);
            sessions[s].next = tok;
            trace[s].push((tok, bits_hash(row)));
        }
    }
    (trace, widths)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(
        (3..=5).contains(&args.len()),
        "usage: dsv4_rows_gate <model-dir> <source.txt> [steps] [timing-steps]"
    );
    let steps: usize = args.get(3).map_or(24, |v| v.parse().expect("steps"));
    let timing_steps: usize = args.get(4).map_or(64, |v| v.parse().expect("timing steps"));
    assert!(steps >= 2);
    let dir = Path::new(&args[1]);
    let tape = SourceTape::read(&args[2]).expect("source tape");
    let tokenizer = Tokenizer::from_hf_dir(dir).expect("tokenizer");
    let prompts: Vec<Vec<u32>> = FRAMES
        .iter()
        .zip(LENS)
        .map(|(frame, len)| tape.prompt(&tokenizer, frame, len)[..len].to_vec())
        .collect();
    let capacity = LENS.iter().max().unwrap() + steps.max(timing_steps) + 128;
    let gpu = Dsv4Gpu::load(dir, &[0, 1], ActQuantVariant::RefFp8Round, capacity).expect("load");
    assert!(!gpu.topology().is_tp_ep(), "PP-2 program only");
    println!(
        "PROTOCOL {{\"sessions\":4,\"prompt_tokens\":{LENS:?},\"join_steps\":{JOIN:?},\"steps\":{steps},\"timing_steps\":{timing_steps},\"greedy\":true,\"compare\":\"full logits bits per step\",\"source_sha256\":\"{}\"}}",
        tape.sha256
    );

    let reference = solo(&gpu, &prompts, capacity, steps);
    let (rows, widths) = batched(&gpu, &prompts, capacity, steps);
    println!("WIDTHS {widths:?}");
    let mut bad = 0usize;
    for s in 0..prompts.len() {
        let first = (0..steps).find(|&i| reference[s][i] != rows[s][i]);
        match first {
            None => println!(
                "SESSION {s} len={} join={} steps={steps} IDENTICAL head={:?}",
                LENS[s],
                JOIN[s],
                &reference[s].iter().take(6).map(|x| x.0).collect::<Vec<_>>()
            ),
            Some(i) => {
                bad += 1;
                println!(
                    "SESSION {s} len={} join={} FIRST DIVERGENCE step={i} solo=(tok {}, bits {:016x}) rows=(tok {}, bits {:016x})",
                    LENS[s],
                    JOIN[s],
                    reference[s][i].0,
                    reference[s][i].1,
                    rows[s][i].0,
                    rows[s][i].1
                );
            }
        }
    }
    if bad > 0 {
        println!("FAILED: {bad} of {} sessions diverged", prompts.len());
        std::process::exit(1);
    }
    println!("PASS: every row bit-identical to its solo step across join, leave and row moves");

    // Timing: one-row steps vs B-row steps, fresh sessions each arm, decode wall only.
    for rep in 0..2 {
        for b in [1usize, 2, 4] {
            let mut sessions: Vec<Session> = prompts[..b]
                .iter()
                .map(|p| prime(&gpu, p, capacity))
                .collect();
            drain(&gpu);
            let t0 = Instant::now();
            for _ in 0..timing_steps {
                for s in &mut sessions {
                    s.next = gpu
                        .decode_step_greedy(s.next, &mut s.state)
                        .expect("one-row");
                }
            }
            let solo_s = t0.elapsed().as_secs_f64();
            let mut sessions: Vec<Session> = prompts[..b]
                .iter()
                .map(|p| prime(&gpu, p, capacity))
                .collect();
            let mut rows = gpu.alloc_rows_state(b).expect("B-row workspace");
            drain(&gpu);
            let t0 = Instant::now();
            for _ in 0..timing_steps {
                let toks: Vec<u32> = sessions.iter().map(|s| s.next).collect();
                let mut states: Vec<&mut DecodeState> =
                    sessions.iter_mut().map(|s| &mut s.state).collect();
                let next = gpu
                    .decode_rows_greedy(&toks, &mut states, &mut rows)
                    .expect("B-row");
                drop(states);
                for (s, t) in sessions.iter_mut().zip(next) {
                    s.next = t;
                }
            }
            let rows_s = t0.elapsed().as_secs_f64();
            let tokens = (b * timing_steps) as f64;
            println!(
                "TIME rep={rep} B={b} one_row_ms_per_token={:.3} rows_ms_per_step={:.3} one_row_tok_s={:.2} rows_tok_s={:.2} speedup={:.3}",
                1e3 * solo_s / tokens,
                1e3 * rows_s / timing_steps as f64,
                tokens / solo_s,
                tokens / rows_s,
                solo_s / rows_s
            );
        }
    }
}
