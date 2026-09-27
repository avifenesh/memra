//! Pinned MiMo audio patch forward on already grouped code IDs. This checks
//! source weight execution, not raw-audio tokenization or serving.

use std::path::Path;

use memra_engine::Engine;
use memra_engine::mimo_audio_patch_load::MiMoAudioPatchWeights;
use memra_gguf::model_packs::mimo_v2::bind_pinned_text_source;
use memra_gguf::source::SafetensorsSource;

type Fail = Box<dyn std::error::Error>;
const CHANNELS: usize = 20;
const GROUP_SIZE: usize = 4;
const OUTPUT: usize = 4_096;

fn run() -> Result<(), Fail> {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .ok_or("usage: mimo_audio_patch_source_probe <source_dir> <gpu>")?;
    let gpu: usize = args
        .next()
        .ok_or("MiMo audio patch source probe needs one GPU ordinal")?
        .parse()?;
    if args.next().is_some() {
        return Err("usage: mimo_audio_patch_source_probe <source_dir> <gpu>".into());
    }
    let source = SafetensorsSource::open(Path::new(&dir))?;
    let (config, _, binding) = bind_pinned_text_source(&source)?;
    let engine = Engine::new(gpu)?;
    let weights = MiMoAudioPatchWeights::load(&engine, &source, &binding, &config)?;
    let codes = (0..GROUP_SIZE * CHANNELS)
        .map(|index| ((index * 17) % 1_280) as u16)
        .collect::<Vec<_>>();
    let output = weights.forward_grouped_codes(&engine, &codes, 1)?;
    let output = engine.dtoh(&output)?;
    if output.len() != OUTPUT || output.iter().any(|value| !value.is_finite()) {
        return Err("MiMo grouped audio output is incomplete or non-finite".into());
    }
    println!("format\tmemra-mimo-audio-patch-source-probe-v1");
    println!("gpu\t{gpu}");
    println!("groups\t1");
    println!("output_width\t{OUTPUT}");
    for (index, value) in output.iter().enumerate() {
        println!("output\t{index}\t{:08x}", value.to_bits());
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("mimo_audio_patch_source_probe: {error}");
        std::process::exit(1);
    }
}
