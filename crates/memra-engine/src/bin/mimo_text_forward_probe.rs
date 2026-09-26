//! Pinned source text-token replay through the model-owned Memra forward.
//! This is a short-context diagnostic, not a server or throughput path.

use std::path::Path;
use std::sync::Arc;

use memra_engine::Engine;
use memra_engine::mimo_text_weights::MiMoTextWeights;
use memra_gguf::source::SafetensorsSource;

type Fail = Box<dyn std::error::Error>;

fn run() -> Result<(), Fail> {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .ok_or("usage: mimo_text_forward_probe <source_dir> <token_id> [token_id ...]")?;
    let token_ids = args
        .map(|text| text.parse::<u32>().map_err(Into::into))
        .collect::<Result<Vec<_>, Fail>>()?;
    if token_ids.is_empty()
        || token_ids.len() > memra_engine::mimo_text_forward::MAX_TEXT_CONTEXT_TOKENS
    {
        return Err("MiMo text replay needs 1..=256 token IDs".into());
    }
    let source = Arc::new(SafetensorsSource::open(Path::new(&dir))?);
    let cards = [Engine::new(0)?, Engine::new(1)?];
    let engines = [&cards[0], &cards[1]];
    let weights = MiMoTextWeights::load(engines, source)?;
    let mut forward = weights.text_forward(engines)?;

    println!("format\tmemra-mimo-model-owned-text-forward-v1");
    let mut logits = forward.token(token_ids[0])?;
    println!("processed_token\t0\t{}", token_ids[0]);
    for (turn, &token) in token_ids.iter().enumerate().skip(1) {
        logits = forward.token(token)?;
        println!("processed_token\t{turn}\t{token}");
    }
    if forward.position() != token_ids.len() || logits.len() != 152_576 {
        return Err("MiMo text replay returned incomplete logits or KV positions".into());
    }
    let mut argmax = 0;
    for index in 1..logits.len() {
        if logits[index] > logits[argmax] {
            argmax = index;
        }
    }
    println!("argmax\t{argmax}");
    for (index, value) in logits.iter().enumerate() {
        println!("logit\t{index}\t{:08x}", value.to_bits());
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("mimo_text_forward_probe: {error}");
        std::process::exit(1);
    }
}
