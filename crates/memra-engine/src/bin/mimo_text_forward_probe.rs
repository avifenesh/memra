//! Pinned source text-token replay through the model-owned Memra forward.
//! This is a short-context diagnostic, not a server or throughput path.

use std::path::Path;
use std::sync::Arc;

use memra_engine::Engine;
use memra_engine::mimo_compressed_text_forward::MiMoCompressedTextForward;
use memra_engine::mimo_text_forward::MiMoTextForward;
use memra_engine::mimo_text_weights::MiMoTextWeights;
use memra_gguf::source::SafetensorsSource;

type Fail = Box<dyn std::error::Error>;
const FOUR_GIB: usize = 4 * 1024 * 1024 * 1024;

enum Forward<'a> {
    Plain(MiMoTextForward<'a>),
    Compressed(Box<MiMoCompressedTextForward<'a>>),
}

impl Forward<'_> {
    fn token(&mut self, token: u32) -> Result<Vec<f32>, Fail> {
        match self {
            Self::Plain(forward) => forward.token(token),
            Self::Compressed(forward) => forward.token(token),
        }
    }

    fn position(&self) -> usize {
        match self {
            Self::Plain(forward) => forward.position(),
            Self::Compressed(forward) => forward.position(),
        }
    }
}

fn run() -> Result<(), Fail> {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .ok_or("usage: mimo_text_forward_probe <source_dir> [--compressed-context=N] <token_id> [token_id ...]")?;
    let first = args
        .next()
        .ok_or("MiMo text replay needs at least one token ID")?;
    let (compressed_context, first_token) =
        if let Some(cap) = first.strip_prefix("--compressed-context=") {
            (
                Some(cap.parse::<usize>()?),
                args.next()
                    .ok_or("MiMo compressed replay has no token ID")?,
            )
        } else {
            (None, first)
        };
    let token_ids = std::iter::once(first_token)
        .chain(args)
        .map(|text| text.parse::<u32>().map_err(Into::into))
        .collect::<Result<Vec<_>, Fail>>()?;
    if token_ids.len() > memra_engine::mimo_text_forward::MAX_TEXT_CONTEXT_TOKENS
        || compressed_context.is_some_and(|cap| token_ids.len() > cap)
    {
        return Err("MiMo text replay needs 1..=256 token IDs".into());
    }
    let source = Arc::new(SafetensorsSource::open(Path::new(&dir))?);
    let cards = [Engine::new(0)?, Engine::new(1)?];
    let engines = [&cards[0], &cards[1]];
    let weights = MiMoTextWeights::load(engines, source)?;
    let mut forward = match compressed_context {
        Some(max) => Forward::Compressed(Box::new(weights.compressed_text_forward(
            engines,
            max,
            [FOUR_GIB; 2],
        )?)),
        None => Forward::Plain(weights.text_forward(engines)?),
    };

    println!(
        "format\t{}",
        if compressed_context.is_some() {
            "memra-mimo-model-owned-compressed-text-forward-v1"
        } else {
            "memra-mimo-model-owned-text-forward-v1"
        }
    );
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
