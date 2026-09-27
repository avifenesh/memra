//! Execute one pinned MiMo vision block on an exact BF16-valued input row
//! archive, for paired publisher-versus-Memra block localization.

use std::path::Path;

use memra_engine::Engine;
use memra_engine::mimo_vision_load::MiMoVisionWeights;
use memra_gguf::model_packs::mimo_v2::bind_pinned_text_source;
use memra_gguf::model_packs::mimo_v2::vision::MiMoVisionGrid;
use memra_gguf::source::SafetensorsSource;
use sha2::{Digest, Sha256};

type Fail = Box<dyn std::error::Error>;
const PATCHES: usize = 4;
const HIDDEN: usize = 1_280;

fn run() -> Result<(), Fail> {
    let mut args = std::env::args().skip(1);
    let source_dir = args
        .next()
        .ok_or("usage: mimo_vision_one_block_input_probe <source_dir> <layer> <input_f32le>")?;
    let layer: usize = args
        .next()
        .ok_or("MiMo vision one-block probe needs a layer")?
        .parse()?;
    let input_path = args
        .next()
        .ok_or("MiMo vision one-block probe needs an input row file")?;
    if args.next().is_some() || layer >= 28 {
        return Err("MiMo vision one-block probe has extra arguments or invalid layer".into());
    }
    let source = SafetensorsSource::open(Path::new(&source_dir))?;
    let (config, _, binding) = bind_pinned_text_source(&source)?;
    let bytes = std::fs::read(&input_path)?;
    if bytes.len() != PATCHES * HIDDEN * 4 {
        return Err("MiMo vision one-block input has an incorrect byte extent".into());
    }
    let input = bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect::<Vec<_>>();
    if input
        .iter()
        .any(|value| !value.is_finite() || value.to_bits() & 0xffff != 0)
    {
        return Err("MiMo vision one-block input is non-finite or not BF16-valued".into());
    }
    let engine = Engine::new(0)?;
    let weights = MiMoVisionWeights::load(&engine, &source, &binding, &config)?;
    let grids = [MiMoVisionGrid {
        frames: 1,
        height: 2,
        width: 2,
    }];
    let uploaded = engine.htod(&input)?;
    let output = weights.forward_one_block(&engine, &config, layer, &grids, &uploaded)?;
    let output = engine.dtoh(&output)?;
    if output.len() != PATCHES * HIDDEN || output.iter().any(|value| !value.is_finite()) {
        return Err("MiMo one-block output is incomplete or non-finite".into());
    }
    println!("format\tmemra-mimo-vision-one-block-input-v1");
    println!("layer\t{layer}");
    println!("input_sha256\t{:x}", Sha256::digest(&bytes));
    for (index, value) in output.iter().enumerate() {
        println!("output\t{index}\t{:08x}", value.to_bits());
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("mimo_vision_one_block_input_probe: {error}");
        std::process::exit(1);
    }
}
