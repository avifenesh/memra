//! Pinned MiMo Conv3D patch projection and all 28 vision blocks on four
//! deterministic, source-shaped pixel patches. This stops before the merger.

use std::path::Path;

use memra_engine::Engine;
use memra_engine::mimo_vision_load::MiMoVisionWeights;
use memra_gguf::model_packs::mimo_v2::bind_pinned_text_source;
use memra_gguf::model_packs::mimo_v2::vision::MiMoVisionGrid;
use memra_gguf::source::SafetensorsSource;
use sha2::{Digest, Sha256};

type Fail = Box<dyn std::error::Error>;
const PATCHES: usize = 4;
const PIXEL_ELEMENTS: usize = 3 * 2 * 16 * 16;
const HIDDEN: usize = 1_280;

fn run_card(
    gpu: usize,
    source: &SafetensorsSource,
    config: &memra_gguf::config::ModelConfig,
    binding: &memra_gguf::checkpoint_binding::CheckpointBinding,
) -> Result<Vec<f32>, Fail> {
    let engine = Engine::new(gpu)?;
    let weights = MiMoVisionWeights::load(&engine, source, binding, config)?;
    let pixels = (0..PATCHES * PIXEL_ELEMENTS)
        .map(|index| ((index * 17 % 251) as f32 - 125.0) / 256.0)
        .collect::<Vec<_>>();
    let input = engine.htod(&pixels)?;
    let grids = [MiMoVisionGrid {
        frames: 1,
        height: 2,
        width: 2,
    }];
    let output = weights.forward_patchified_blocks(&engine, config, &grids, &input)?;
    let output = engine.dtoh(&output)?;
    if output.len() != PATCHES * HIDDEN || output.iter().any(|value| !value.is_finite()) {
        return Err("MiMo vision blocks returned incomplete or non-finite patch rows".into());
    }
    Ok(output)
}

fn run() -> Result<(), Fail> {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .ok_or("usage: mimo_vision_blocks_source_probe <source_dir>")?;
    if args.next().is_some() {
        return Err("usage: mimo_vision_blocks_source_probe <source_dir>".into());
    }
    let source = SafetensorsSource::open(Path::new(&dir))?;
    let (config, _, binding) = bind_pinned_text_source(&source)?;
    let first = run_card(0, &source, &config, &binding)?;
    let second = run_card(1, &source, &config, &binding)?;
    if first
        .iter()
        .zip(&second)
        .any(|(left, right)| left.to_bits() != right.to_bits())
    {
        return Err("MiMo vision blocks differ between the two RTX PRO 6000 cards".into());
    }
    let mut hash = Sha256::new();
    for value in &first {
        hash.update(value.to_bits().to_le_bytes());
    }
    println!("format\tmemra-mimo-vision-blocks-source-probe-v1");
    println!("input\tdeterministic_four_patch_pixels");
    println!("cards\t0,1");
    println!("patches\t{PATCHES}");
    println!("blocks\t28");
    println!("output_width\t{HIDDEN}");
    println!("output_sha256\t{:x}", hash.finalize());
    for (index, value) in first.iter().enumerate() {
        println!("output\t{index}\t{:08x}", value.to_bits());
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("mimo_vision_blocks_source_probe: {error}");
        std::process::exit(1);
    }
}
