//! Pinned bundled MiMo audio-tokenizer 24-layer encoder on two prepared
//! BF16-valued rows. Prints the post-skip, post-final-norm output for an
//! independent publisher comparison; no PCM, downsample, or RVQ is run.

use std::path::Path;

use memra_engine::Engine;
use memra_engine::mimo_audio_codec_weights::MiMoAudioCodecEncoderWeights;
use memra_reference::mimo_audio_codec_layer::bf16;
use sha2::{Digest, Sha256};

type Fail = Box<dyn std::error::Error>;
const TOKENS: usize = 2;
const HIDDEN: usize = 1_024;

fn run() -> Result<(), Fail> {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .ok_or("usage: mimo_audio_stack_source_probe <source_dir> <gpu>")?;
    let gpu: usize = args
        .next()
        .ok_or("MiMo audio stack source probe needs a GPU ordinal")?
        .parse()?;
    if args.next().is_some() || gpu > 1 {
        return Err("MiMo audio stack source probe has extra arguments or wrong GPU".into());
    }
    let engine = Engine::new(gpu)?;
    let weights = MiMoAudioCodecEncoderWeights::load(&engine, Path::new(&dir))?;
    let input = (0..TOKENS * HIDDEN)
        .map(|index| bf16((index as f32 % 17.0 - 8.0) * 0.015625))
        .collect::<Vec<_>>();
    let rows = weights.encode_transformer_stack(&engine, &engine.htod(&input)?, TOKENS)?;
    let rows = engine.dtoh(&rows)?;
    if rows.len() != TOKENS * HIDDEN || rows.iter().any(|value| !value.is_finite()) {
        return Err("MiMo audio encoder stack returned incomplete or non-finite rows".into());
    }
    let mut hash = Sha256::new();
    for value in &rows {
        hash.update(value.to_bits().to_le_bytes());
    }
    println!("format\tmemra-mimo-audio-stack-source-probe-v1");
    println!("gpu\t{gpu}");
    println!("tokens\t{TOKENS}");
    println!("width\t{HIDDEN}");
    println!("output_sha256\t{:x}", hash.finalize());
    for (index, value) in rows.iter().enumerate() {
        println!("output\t{index}\t{:08x}", value.to_bits());
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("mimo_audio_stack_source_probe: {error}");
        std::process::exit(1);
    }
}
