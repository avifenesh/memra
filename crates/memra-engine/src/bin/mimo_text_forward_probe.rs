//! Pinned source text-token replay through the model-owned Memra forward.
//! This is a short-context diagnostic, not a server or throughput path.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

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

fn argmax(logits: &[f32]) -> usize {
    let mut best = 0;
    for index in 1..logits.len() {
        if logits[index].total_cmp(&logits[best]).is_gt() {
            best = index;
        }
    }
    best
}

fn run() -> Result<(), Fail> {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .ok_or("usage: mimo_text_forward_probe <source_dir> [--compressed-context=N | --s5-g16-context=N] <token_id> [token_id ...]")?;
    let first = args
        .next()
        .ok_or("MiMo text replay needs at least one token ID")?;
    let (compressed_context, s5_g16, first_token) =
        if let Some(cap) = first.strip_prefix("--compressed-context=") {
            (
                Some(cap.parse::<usize>()?),
                false,
                args.next()
                    .ok_or("MiMo compressed replay has no token ID")?,
            )
        } else if let Some(cap) = first.strip_prefix("--s5-g16-context=") {
            (
                Some(cap.parse::<usize>()?),
                true,
                args.next().ok_or("MiMo S5 replay has no token ID")?,
            )
        } else {
            (None, false, first)
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
    if s5_g16 && compressed_context.is_none_or(|cap| !(1..=256).contains(&cap)) {
        return Err("MiMo S5 diagnostic context must contain 1..=256 tokens".into());
    }
    let source = Arc::new(SafetensorsSource::open(Path::new(&dir))?);
    let cards = [Engine::new(0)?, Engine::new(1)?];
    let engines = [&cards[0], &cards[1]];
    let weights = MiMoTextWeights::load(engines, source)?;
    let mut forward = match compressed_context {
        Some(max) => Forward::Compressed(Box::new(if s5_g16 {
            weights.compressed_text_forward_s5_g16(engines, max, [FOUR_GIB; 2])?
        } else {
            weights.compressed_text_forward(engines, max, [FOUR_GIB; 2])?
        })),
        None => Forward::Plain(weights.text_forward(engines)?),
    };

    println!(
        "format\t{}",
        if s5_g16 {
            "memra-mimo-model-owned-s5-g16-text-forward-experiment-v1"
        } else if compressed_context.is_some() {
            "memra-mimo-model-owned-compressed-text-forward-v1"
        } else {
            "memra-mimo-model-owned-text-forward-v1"
        }
    );
    let turn_start = Instant::now();
    let mut logits = forward.token(token_ids[0])?;
    println!("processed_token\t0\t{}", token_ids[0]);
    let mut last_argmax = None;
    if compressed_context.is_some() {
        let top = argmax(&logits);
        last_argmax = Some(top);
        println!("argmax_turn\t0\t{top}");
        println!(
            "turn_wall_ms\t0\t{:.3}",
            turn_start.elapsed().as_secs_f64() * 1000.0
        );
    }
    for (turn, &token) in token_ids.iter().enumerate().skip(1) {
        let turn_start = Instant::now();
        logits = forward.token(token)?;
        println!("processed_token\t{turn}\t{token}");
        if compressed_context.is_some() {
            let top = argmax(&logits);
            last_argmax = Some(top);
            println!("argmax_turn\t{turn}\t{top}");
            println!(
                "turn_wall_ms\t{turn}\t{:.3}",
                turn_start.elapsed().as_secs_f64() * 1000.0
            );
        }
    }
    if forward.position() != token_ids.len() || logits.len() != 152_576 {
        return Err("MiMo text replay returned incomplete logits or KV positions".into());
    }
    println!("argmax\t{}", last_argmax.unwrap_or_else(|| argmax(&logits)));
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
