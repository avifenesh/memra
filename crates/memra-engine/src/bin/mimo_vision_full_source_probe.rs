//! Pinned MiMo Conv3D projection, all 28 vision blocks, and the publisher
//! merger on four deterministic source-shaped pixel patches. This offline
//! component stops at modal rows before image/video request processing.

use std::path::Path;

use memra_engine::Engine;
use memra_engine::mimo_vision_load::MiMoVisionWeights;
use memra_gguf::checkpoint_binding::CheckpointBinding;
use memra_gguf::config::ModelConfig;
use memra_gguf::model_packs::mimo_v2::bind_pinned_text_source;
use memra_gguf::model_packs::mimo_v2::vision::MiMoVisionGrid;
use memra_gguf::source::SafetensorsSource;
use sha2::{Digest, Sha256};

type Fail = Box<dyn std::error::Error>;
const PATCHES: usize = 4;
const PIXEL_ELEMENTS: usize = 3 * 2 * 16 * 16;
const OUTPUT: usize = 4_096;

fn run_card(
    gpu: usize,
    source: &SafetensorsSource,
    config: &ModelConfig,
    binding: &CheckpointBinding,
) -> Result<Vec<f32>, Fail> {
    let engine = Engine::new(gpu)?;
    let weights = MiMoVisionWeights::load(&engine, source, binding, config)?;
    let pixels = (0..PATCHES * PIXEL_ELEMENTS)
        .map(|index| ((index * 17 % 251) as f32 - 125.0) / 256.0)
        .collect::<Vec<_>>();
    let grids = [MiMoVisionGrid {
        frames: 1,
        height: 2,
        width: 2,
    }];
    let input = engine.htod(&pixels)?;
    let output = weights.forward_patchified_vision(&engine, config, &grids, &input)?;
    let output = engine.dtoh(&output)?;
    if output.len() != OUTPUT
        || output
            .iter()
            .any(|value| !value.is_finite() || value.to_bits() & 0xffff != 0)
    {
        return Err("MiMo full visual returned incomplete or non-BF16 modal rows".into());
    }
    Ok(output)
}

fn run() -> Result<(), Fail> {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .ok_or("usage: mimo_vision_full_source_probe <source_dir>")?;
    if args.next().is_some() {
        return Err("usage: mimo_vision_full_source_probe <source_dir>".into());
    }
    let source = SafetensorsSource::open(Path::new(&dir))?;
    let (config, _, binding) = bind_pinned_text_source(&source)?;
    let first = run_card(0, &source, &config, &binding)?;
    let second = run_card(1, &source, &config, &binding)?;
    if first
        .iter()
        .zip(second.iter())
        .any(|(left, right)| left.to_bits() != right.to_bits())
    {
        return Err("MiMo full visual output differs between the two cards".into());
    }
    let mut hash = Sha256::new();
    for value in &first {
        hash.update(value.to_bits().to_le_bytes());
    }
    println!("format\tmemra-mimo-full-vision-source-probe-v1");
    println!("input\tdeterministic_four_patch_pixels");
    println!("cards\t0,1");
    println!("patches\t{PATCHES}");
    println!("output_rows\t1");
    println!("output_width\t{OUTPUT}");
    println!("output_sha256\t{:x}", hash.finalize());
    for (index, value) in first.iter().enumerate() {
        println!("output\t{index}\t{:08x}", value.to_bits());
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("mimo_vision_full_source_probe: {error}");
        std::process::exit(1);
    }
}
