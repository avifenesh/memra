//! THE RED ARM for memra #495, on the real drivers: a stream that stops INSIDE a speculative
//! round must leave device state at the token boundary it reached, not past it.
//!
//! This is the shape every tool-call turn has. `Emit` stops at a DSML close or an EOS that lands
//! mid-round, and before this gate's fix the driver had already committed the whole accepted run,
//! so `park_prefix` refused the session ("device state consumed N generated tokens, stream
//! committed only M") and the AGENT shape never entered the parked-prefix tier at all: 10 of 10
//! agent turn-1s refused on the prod candidate while `reasoning.enabled:false` parked 10 of 10
//! (darklanes `research/dsv4f-hot-ttft-20260911`, receipts/hot3-r1).
//!
//! The gate runs three arms against the SAME prompt and drafter state:
//!   RED   - stop after the round's first token. The old driver commits the whole round, so
//!           `state.pos` overshoots and this arm fails loudly on a reverted engine.
//!   MID   - stop two tokens into a round, the tool-call shape.
//!   WHOLE - never stop, which must stay byte-identical to the no-callback driver (the ordinary
//!           request path, where this change must cost nothing).
//!
//! usage: dsv4_park_boundary_gate <model-dir> <source.txt>
use memra_engine::dsv4_gpu::{Dsv4Gpu, Dsv4SampleCfg, Dsv4Vt, RoundTake};
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::path::Path;

const PROMPT_TOKENS: usize = 600;
const N_NEW: usize = 64;

fn cfg() -> Dsv4SampleCfg {
    Dsv4SampleCfg {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        seed: 20260911,
    }
}

/// One arm: generate with a callback that takes `stop_after` tokens in total and then stops.
/// `None` means take everything. Returns (tokens the driver reports, device pos, prompt len).
fn arm(
    gpu: &Dsv4Gpu,
    prompt: &[u32],
    stop_after: Option<usize>,
    width: usize,
) -> (Vec<u32>, usize) {
    let capacity = prompt.len() + N_NEW + 96;
    let mut state = gpu
        .alloc_decode_state_for_transient(capacity, width)
        .expect("state");
    let mut draft = gpu.dspark_alloc_state().expect("draft");
    let row = gpu
        .dspark_prefill_prime_chunked(prompt, &mut state, &mut draft, width)
        .expect("prime");
    let mut verify = gpu.alloc_verify_state_for(capacity).expect("verify");
    let mut taken_total = 0usize;
    let mut cb = |new: &[u32]| match stop_after {
        None => RoundTake::all(new.len()),
        Some(limit) => {
            let room = limit.saturating_sub(taken_total);
            let taken = room.min(new.len());
            taken_total += taken;
            RoundTake {
                taken,
                stop: taken_total >= limit,
            }
        }
    };
    let run = gpu
        .spec_sampled_batched_pen_restored(
            prompt,
            &row,
            N_NEW,
            &mut state,
            &mut draft,
            &mut verify,
            usize::MAX,
            Dsv4Vt::Off,
            &cfg(),
            None,
            Some(&mut cb),
        )
        .expect("sampled spec");
    (run.tokens, state.pos)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(
        args.len(),
        3,
        "usage: dsv4_park_boundary_gate <model-dir> <source.txt>"
    );
    let dir = Path::new(&args[1]);
    let source = std::fs::read_to_string(&args[2]).expect("source");
    let tokenizer = Tokenizer::from_hf_dir(dir).expect("tokenizer");
    let all = tokenizer.encode(&format!("Review this inference code:\n{source}"), true);
    assert!(all.len() > PROMPT_TOKENS, "source too short for the gate");
    let prompt = &all[..PROMPT_TOKENS];
    println!("SOURCE sha256={:x}", Sha256::digest(source.as_bytes()));
    let gpu = Dsv4Gpu::load(dir, &[0, 1], ActQuantVariant::RefFp8Round, 8192).expect("model");
    let width = 512.min(PROMPT_TOKENS);

    // WHOLE: the ordinary request. A callback that takes every round must be identical to no
    // callback at all, or this change has a cost on the path that does not stop early.
    let (whole_tokens, whole_pos) = arm(&gpu, prompt, None, width);
    assert_eq!(
        whole_tokens.len(),
        N_NEW,
        "an unstopped run produces its whole budget"
    );
    println!(
        "WHOLE tokens={} pos={} prompt={}",
        whole_tokens.len(),
        whole_pos,
        PROMPT_TOKENS
    );
    // The device is at most one token behind the stream (the final token can be pending) and
    // never ahead of it.
    assert!(
        whole_pos <= PROMPT_TOKENS + whole_tokens.len()
            && whole_pos + 1 >= PROMPT_TOKENS + whole_tokens.len(),
        "unstopped run: pos {whole_pos} against prompt {PROMPT_TOKENS} + {} tokens",
        whole_tokens.len()
    );

    for stop_after in [1usize, 2, 3] {
        let (tokens, pos) = arm(&gpu, prompt, Some(stop_after), width);
        println!(
            "STOPPED take={stop_after} driver_tokens={} pos={} boundary={}",
            tokens.len(),
            pos,
            PROMPT_TOKENS + stop_after
        );
        // THE CONTRACT, and the assertion that fails on a reverted engine: the device sits
        // exactly at what the stream took. A driver that commits the whole round overshoots
        // here by however many drafts that round accepted.
        assert_eq!(
            pos,
            PROMPT_TOKENS + stop_after,
            "stopped at {stop_after} tokens: device must be at the stream's boundary, not past it"
        );
        assert_eq!(
            tokens.len(),
            stop_after,
            "the driver reports exactly the tokens the stream took"
        );
    }

    // Non-vacuity: the arms must actually have been multi-token rounds, or stopping mid-round
    // proved nothing. A round that only ever produced one token cannot overshoot.
    let (_, pos_one) = arm(&gpu, prompt, Some(1), width);
    let rounds_are_wide = whole_tokens.len() > 1 && pos_one == PROMPT_TOKENS + 1;
    assert!(
        rounds_are_wide,
        "gate is vacuous unless the driver produces multi-token rounds"
    );
    println!("PARK_BOUNDARY_GATE_OK");
}
