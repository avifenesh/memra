//! PP-2 request pipelining gate (memra #667).
//!
//! Two sessions decode the same greedy streams twice: serially, one session after the other
//! through `decode_step_greedy`, then pipelined through `decode_step_greedy_enqueue`, `_wait` and
//! `_complete` in the order a two-session serve loop takes: queue A, queue
//! B, then finish A and queue its next step, finish B and queue its next. Each stage stream
//! then holds the two sessions' steps in order, so card 0 runs one session's stage 0 while
//! card 1 runs the other's stage 1.
//!
//! Correctness: every token of both sessions equals its serial stream, on every repeat.
//! Timing: decode wall time only (both prefills complete and drain first), aggregate
//! tokens per second for each arm, serial and pipelined repeats interleaved.
//!
//! Usage: `dsv4_pipeline_gate <model-dir> <source.txt> [output-tokens] [repeats]`.
//! Rig law: under the box GPU lock, served defaults (no MEMRA_DSV4_* overrides).
use memra_engine::dsv4_gpu::{DecodeState, Dsv4Gpu};
use memra_engine::dsv4_source_tape::SourceTape;
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use std::{path::Path, time::Instant};

const PROMPT: usize = 256;
/// The served matrix program primes through chunked prefill (the monolithic API refuses it).
const CHUNK: usize = 256;

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

struct Session {
    state: DecodeState,
    tokens: Vec<u32>,
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
        tokens: vec![argmax(&logits)],
    }
}

fn serial(
    gpu: &Dsv4Gpu,
    prompts: &[Vec<u32>; 2],
    capacity: usize,
    output: usize,
) -> (Vec<Vec<u32>>, f64) {
    let mut sessions: Vec<Session> = prompts.iter().map(|p| prime(gpu, p, capacity)).collect();
    drain(gpu);
    let t0 = Instant::now();
    for session in &mut sessions {
        while session.tokens.len() < output {
            let last = *session.tokens.last().unwrap();
            let next = gpu
                .decode_step_greedy(last, &mut session.state)
                .expect("serial step");
            session.tokens.push(next);
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    (sessions.into_iter().map(|s| s.tokens).collect(), secs)
}

fn pipelined(
    gpu: &Dsv4Gpu,
    prompts: &[Vec<u32>; 2],
    capacity: usize,
    output: usize,
) -> (Vec<Vec<u32>>, f64) {
    let mut sessions: Vec<Session> = prompts.iter().map(|p| prime(gpu, p, capacity)).collect();
    drain(gpu);
    let t0 = Instant::now();
    for session in &mut sessions {
        let last = *session.tokens.last().unwrap();
        gpu.decode_step_greedy_enqueue(last, &mut session.state)
            .expect("first enqueue");
    }
    while sessions.iter().any(|s| s.tokens.len() < output) {
        for session in &mut sessions {
            if session.tokens.len() >= output {
                continue;
            }
            gpu.decode_step_greedy_wait(&mut session.state)
                .expect("pipelined wait");
            let next = gpu
                .decode_step_greedy_complete(&mut session.state)
                .expect("pipelined complete");
            session.tokens.push(next);
            if session.tokens.len() < output {
                gpu.decode_step_greedy_enqueue(next, &mut session.state)
                    .expect("pipelined enqueue");
            }
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    (sessions.into_iter().map(|s| s.tokens).collect(), secs)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(
        (3..=5).contains(&args.len()),
        "usage: dsv4_pipeline_gate <model-dir> <source.txt> [output-tokens] [repeats]"
    );
    let output: usize = args
        .get(3)
        .map_or(256, |v| v.parse().expect("output tokens"));
    let repeats: usize = args.get(4).map_or(3, |v| v.parse().expect("repeats"));
    assert!(output >= 2 && repeats >= 1);
    let dir = Path::new(&args[1]);
    let tape = SourceTape::read(&args[2]).expect("source tape");
    let tokenizer = Tokenizer::from_hf_dir(dir).expect("tokenizer");
    // `SourceTape::prompt` returns the whole tokenized tape; each session takes its head.
    let prompts = [
        tape.prompt(
            &tokenizer,
            "Review this inference engine source:\n\n",
            PROMPT,
        )[..PROMPT]
            .to_vec(),
        tape.prompt(
            &tokenizer,
            "Explain what this inference engine source does:\n\n",
            PROMPT,
        )[..PROMPT]
            .to_vec(),
    ];
    assert_ne!(
        prompts[0], prompts[1],
        "the two sessions need distinct prompts"
    );
    let capacity = prompts.iter().map(Vec::len).max().unwrap() + output + 96;
    println!(
        "PROTOCOL {{\"sessions\":2,\"prompt_tokens\":[{},{}],\"output_tokens\":{output},\"repeats\":{repeats},\"greedy\":true,\"order\":\"serial,pipelined per repeat\",\"timing\":\"decode wall after drained prefills\",\"source_sha256\":\"{}\"}}",
        prompts[0].len(),
        prompts[1].len(),
        tape.sha256
    );
    let gpu = Dsv4Gpu::load(dir, &[0, 1], ActQuantVariant::RefFp8Round, capacity).expect("model");
    assert!(!gpu.topology().is_tp_ep(), "PP-2 program only");
    let (reference, _) = serial(&gpu, &prompts, capacity, output);
    for (i, tokens) in reference.iter().enumerate() {
        println!(
            "SESSION {i} tokens={} head={:?}",
            tokens.len(),
            &tokens[..8.min(tokens.len())]
        );
    }
    let decoded = 2 * (output - 1);
    for repeat in 0..repeats {
        let (serial_tokens, serial_secs) = serial(&gpu, &prompts, capacity, output);
        assert_eq!(
            serial_tokens, reference,
            "serial repeat {repeat} changed its stream"
        );
        let (pipe_tokens, pipe_secs) = pipelined(&gpu, &prompts, capacity, output);
        for (i, (got, want)) in pipe_tokens.iter().zip(&reference).enumerate() {
            if let Some(at) = got.iter().zip(want).position(|(a, b)| a != b) {
                panic!(
                    "session {i} pipelined token {at}: {} != serial {}",
                    got[at], want[at]
                );
            }
            assert_eq!(got.len(), want.len(), "session {i} pipelined length");
        }
        println!(
            "TIME repeat={repeat} serial_s={serial_secs:.4} pipelined_s={pipe_secs:.4} serial_tok_s={:.2} pipelined_tok_s={:.2} speedup={:.3} identical=true",
            decoded as f64 / serial_secs,
            decoded as f64 / pipe_secs,
            serial_secs / pipe_secs
        );
    }
    println!(
        "PASS pipelined greedy == serial greedy, 2 sessions x {output} tokens x {repeats} repeats"
    );
}
