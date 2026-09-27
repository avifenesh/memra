//! Pinned source MiMo patch projection and all 28 vision blocks with a
//! layerwise row receipt for comparison with the publisher's premerger tower.

use std::path::Path;

use cudarc::driver::CudaSlice;
use memra_engine::Engine;
use memra_engine::mimo_vision_load::MiMoVisionWeights;
use memra_gguf::model_packs::mimo_v2::bind_pinned_text_source;
use memra_gguf::model_packs::mimo_v2::vision::MiMoVisionGrid;
use memra_gguf::source::SafetensorsSource;

type Fail = Box<dyn std::error::Error>;
const PATCHES: usize = 4;
const PIXEL_ELEMENTS: usize = 3 * 2 * 16 * 16;
const HIDDEN: usize = 1_280;

fn emit_layer(engine: &Engine, name: &str, hidden: &CudaSlice<f32>) -> Result<(), Fail> {
    let rows = engine.dtoh(hidden)?;
    if rows.len() != PATCHES * HIDDEN || rows.iter().any(|value| !value.is_finite()) {
        return Err(format!("MiMo vision {name} row is incomplete or non-finite").into());
    }
    for (index, value) in rows.iter().enumerate() {
        println!("layer_value\t{name}\t{index}\t{:08x}", value.to_bits());
    }
    Ok(())
}

fn run() -> Result<(), Fail> {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .ok_or("usage: mimo_vision_layerwise_source_probe <source_dir>")?;
    if args.next().is_some() {
        return Err("usage: mimo_vision_layerwise_source_probe <source_dir>".into());
    }
    let source = SafetensorsSource::open(Path::new(&dir))?;
    let (config, _, binding) = bind_pinned_text_source(&source)?;
    let engine = Engine::new(0)?;
    let weights = MiMoVisionWeights::load(&engine, &source, &binding, &config)?;
    let pixels = (0..PATCHES * PIXEL_ELEMENTS)
        .map(|index| ((index * 17 % 251) as f32 - 125.0) / 256.0)
        .collect::<Vec<_>>();
    let grids = [MiMoVisionGrid {
        frames: 1,
        height: 2,
        width: 2,
    }];
    let input = engine.htod(&pixels)?;
    let mut hidden =
        engine.mimo_vision_patch_project(&weights.patch_projection, &input, PATCHES)?;
    println!("format\tmemra-mimo-vision-layerwise-source-v1");
    println!("input\tdeterministic_four_patch_pixels");
    emit_layer(&engine, "patch", &hidden)?;
    for layer in 0..28 {
        hidden = weights.forward_one_block(&engine, &config, layer, &grids, &hidden)?;
        emit_layer(&engine, &format!("block{layer}"), &hidden)?;
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("mimo_vision_layerwise_source_probe: {error}");
        std::process::exit(1);
    }
}
