//! Multi-row identity gate for the all-layer DSV4 TP/EP walk (memra #454).
//!
//! One real-source tape, three arms on fresh request states, every logit row compared bit
//! for bit against the sequential arm at the same position:
//!
//! - SEQ: one-token prime, then every tape token through `decode_step`, the eager walk.
//! - CHUNK: `prefill_with_cache_chunked` over the prompt at width 16, then `decode_step`
//!   over the rest of the tape.
//! - VERIFY: sequential prime of the prompt, then teacher-forced verify rounds of width
//!   1..=6 at tmax 6, each committing a varied prefix, the shape a DSpark round takes. A
//!   short commit rolls the rest of the round back, so the next round's rows also check the
//!   rollback.
//!
//! `tpep` runs the arms on the TP/EP walk, `pp` on the PP-2 program as the control: an
//! inequality both programs show is a property of the shared kernels, one only TP/EP shows
//! is a TP/EP defect. The two programs are different numeric classes (the TP/EP joins sum
//! rank partials), so rows are never compared across programs.

use memra_engine::dsv4_gpu::{DecodeState, Dsv4Gpu};
use memra_engine::dsv4_source_tape::SourceTape;
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use std::path::Path;

const PROMPT_TOKENS: usize = 160;
const TAPE_TOKENS: usize = 224;
const CHUNK: usize = 16;
const TMAX: usize = 6;
/// Round widths and committed prefixes: full rounds, a one-row round, short commits that roll
/// rows back, a round that ends on the C4 and C128 boundaries.
const ROUNDS: &[(usize, usize)] = &[
    (6, 6),
    (1, 1),
    (6, 2),
    (3, 3),
    (6, 1),
    (5, 4),
    (6, 6),
    (2, 1),
    (6, 3),
    (4, 4),
    (6, 5),
    (6, 6),
    (1, 1),
    (6, 6),
];

fn bits(row: &[f32]) -> Vec<u32> {
    assert!(
        row.iter().all(|v| v.is_finite()),
        "non-finite logit in a gate row"
    );
    row.iter().map(|v| v.to_bits()).collect()
}

fn first_diff(a: &[u32], b: &[u32]) -> Option<(usize, f32, f32)> {
    a.iter()
        .zip(b)
        .position(|(x, y)| x != y)
        .map(|i| (i, f32::from_bits(a[i]), f32::from_bits(b[i])))
}

fn argmax(row: &[u32]) -> usize {
    let mut best = 0usize;
    for i in 1..row.len() {
        if f32::from_bits(row[i]) > f32::from_bits(row[best]) {
            best = i;
        }
    }
    best
}

fn fresh(gpu: &Dsv4Gpu, transient: usize) -> DecodeState {
    gpu.alloc_decode_state_for_transient(TAPE_TOKENS + 8, transient)
        .expect("decode state")
}

/// Row `i` is the logits after tape token `i`.
fn seq_rows(gpu: &Dsv4Gpu, tape: &[u32]) -> Vec<Vec<u32>> {
    let mut state = fresh(gpu, TMAX);
    let mut rows = Vec::with_capacity(tape.len());
    rows.push(bits(
        &gpu.prefill_with_cache_chunked(&tape[..1], &mut state, 1)
            .expect("SEQ prime"),
    ));
    for &tok in &tape[1..] {
        rows.push(bits(&gpu.decode_step(tok, &mut state).expect("SEQ step")));
    }
    assert_eq!(state.pos, tape.len());
    rows
}

fn compare(arm: &str, pos: usize, got: &[u32], want: &[u32], fails: &mut Vec<String>) {
    if let Some((col, g, w)) = first_diff(got, want) {
        fails.push(format!(
            "{arm} row {pos}: first diff col {col} got {g:e} want {w:e}, argmax {} vs {}",
            argmax(got),
            argmax(want)
        ));
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(
        args.len() == 4 && (args[3] == "tpep" || args[3] == "pp"),
        "usage: dsv4_tp_ep_verify_gate <model-dir> <real-source.txt> <tpep|pp>"
    );
    let tp_ep = args[3] == "tpep";
    assert_eq!(
        std::env::var("MEMRA_DSV4_DECODE_PATH").as_deref(),
        Ok("device"),
        "requires MEMRA_DSV4_DECODE_PATH=device"
    );
    assert!(
        matches!(
            std::env::var("MEMRA_DSV4_DRAFTER").as_deref(),
            Err(_) | Ok("") | Ok("off")
        ),
        "the gate drives verify rounds itself; the drafter stays unloaded"
    );
    let attention_tp = match std::env::var("MEMRA_DSV4_ATTENTION_TP_GATE").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("0") => false,
        Ok("1") => true,
        _ => panic!("MEMRA_DSV4_ATTENTION_TP_GATE requires 0 or 1"),
    };
    let dir = Path::new(&args[1]);
    let source = SourceTape::read(&args[2]).expect("source tape");
    let tokenizer = Tokenizer::from_hf_dir(dir).expect("tokenizer");
    let tape: Vec<u32> = source
        .prompt(
            &tokenizer,
            "Review this inference engine source:\n\n",
            TAPE_TOKENS,
        )
        .into_iter()
        .take(TAPE_TOKENS)
        .collect();
    let committed: usize = ROUNDS.iter().map(|&(_, n)| n).sum();
    assert!(PROMPT_TOKENS + committed + TMAX <= TAPE_TOKENS);

    Dsv4Gpu::set_tp_ep_topology_for_gate(tp_ep);
    Dsv4Gpu::set_attention_tp_for_gate(tp_ep && attention_tp);
    println!(
        "PROTOCOL {{\"topology\":\"{}\",\"attention_tp\":{},\"prompt\":{PROMPT_TOKENS},\"tape\":{TAPE_TOKENS},\"chunk\":{CHUNK},\"tmax\":{TMAX},\"rounds\":{ROUNDS:?},\"source_sha256\":\"{}\"}}",
        args[3],
        tp_ep && attention_tp,
        source.sha256
    );
    let gpu = Dsv4Gpu::load(dir, &[0, 1], ActQuantVariant::RefFp8Round, 512).expect("load");
    assert_eq!(
        gpu.topology().is_tp_ep(),
        tp_ep,
        "no silent topology fallback"
    );
    assert_eq!(
        gpu.attention_tp_geometry().is_some(),
        tp_ep && attention_tp,
        "no silent attention fallback"
    );
    let calls0 = gpu.tp_ep_rank_layer_calls();

    let seq = seq_rows(&gpu, &tape);
    let seq_again = seq_rows(&gpu, &tape);
    let mut fails = Vec::new();
    for (pos, (a, b)) in seq.iter().zip(&seq_again).enumerate() {
        compare("SEQ-repeat", pos, a, b, &mut fails);
    }
    println!("SEQ rows={} deterministic={}", seq.len(), fails.is_empty());

    // CHUNK: the chunked prime returns the row after its final token.
    let mut chunk_rows = 0usize;
    {
        let mut state = fresh(&gpu, CHUNK);
        let row = bits(
            &gpu.prefill_with_cache_chunked(&tape[..PROMPT_TOKENS], &mut state, CHUNK)
                .expect("CHUNK prime"),
        );
        compare(
            "CHUNK-prime",
            PROMPT_TOKENS - 1,
            &row,
            &seq[PROMPT_TOKENS - 1],
            &mut fails,
        );
        chunk_rows += 1;
        for (pos, &tok) in tape.iter().enumerate().skip(PROMPT_TOKENS) {
            let row = bits(&gpu.decode_step(tok, &mut state).expect("CHUNK step"));
            compare("CHUNK-step", pos, &row, &seq[pos], &mut fails);
            chunk_rows += 1;
        }
    }
    println!("CHUNK rows={chunk_rows}");

    // VERIFY: sequential prime, then teacher-forced rounds.
    let mut verify_rows = 0usize;
    {
        let mut state = fresh(&gpu, TMAX);
        gpu.prefill_with_cache_chunked(&tape[..1], &mut state, 1)
            .expect("VERIFY prime");
        for &tok in &tape[1..PROMPT_TOKENS] {
            gpu.decode_step(tok, &mut state).expect("VERIFY prime step");
        }
        let mut vstate = gpu
            .alloc_verify_state_width_for_gate(state.capacity, TMAX)
            .expect("verify state");
        for &(width, commit) in ROUNDS {
            let pos0 = state.pos;
            let toks = &tape[pos0..pos0 + width];
            let (logits, _) = gpu
                .verify_batch_dev(toks, &mut state, &mut vstate, None, true)
                .expect("VERIFY round");
            let logits = logits.expect("full verify logits");
            let vocab = logits.len() / width;
            for i in 0..width {
                let row = bits(&logits[i * vocab..(i + 1) * vocab]);
                compare(
                    &format!("VERIFY-w{width}c{commit}"),
                    pos0 + i,
                    &row,
                    &seq[pos0 + i],
                    &mut fails,
                );
                verify_rows += 1;
            }
            gpu.commit_verify_dev(&mut state, &mut vstate, commit)
                .expect("VERIFY commit");
            assert_eq!(state.pos, pos0 + commit);
        }
        // Plain steps after the last round read the committed caches.
        for pos in state.pos..state.pos + TMAX {
            let row = bits(&gpu.decode_step(tape[pos], &mut state).expect("VERIFY tail"));
            compare("VERIFY-tail", pos, &row, &seq[pos], &mut fails);
            verify_rows += 1;
        }
    }
    println!("VERIFY rows={verify_rows}");
    if tp_ep {
        let calls = gpu.tp_ep_rank_layer_calls();
        assert!(
            calls[0] > calls0[0] && calls[1] - calls0[1] == calls[0] - calls0[0],
            "both TP/EP ranks must run every layer"
        );
        assert_eq!(gpu.tp_ep_ar_refusal_words().expect("refusal words"), [0, 0]);
    }
    Dsv4Gpu::set_attention_tp_for_gate(false);
    Dsv4Gpu::set_tp_ep_topology_for_gate(false);
    for fail in fails.iter().take(40) {
        println!("FAIL {fail}");
    }
    if fails.is_empty() {
        println!(
            "PASS topology={} rows chunk={chunk_rows} verify={verify_rows} bit-equal to sequential",
            args[3]
        );
    } else {
        println!(
            "FAILED topology={} mismatched_rows={}",
            args[3],
            fails.len()
        );
        std::process::exit(1);
    }
}
