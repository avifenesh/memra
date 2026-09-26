//! Diagnostic full-checkpoint load on exactly two GPUs. This proves resident
//! weight placement and reports live memory, not a token or serving result.

use std::path::Path;
use std::sync::Arc;

use memra_engine::Engine;
use memra_engine::mimo_audio_patch_load::MiMoAudioPatchWeights;
use memra_engine::mimo_mtp_weights::Mtp3Weights;
use memra_engine::mimo_text_weights::MiMoTextWeights;
use memra_engine::mimo_vision_load::MiMoVisionWeights;
use memra_gguf::model_packs::mimo_v2::bind_pinned_text_source;
use memra_gguf::source::SafetensorsSource;

type Fail = Box<dyn std::error::Error>;

fn memory(engine: &Engine) -> Result<(usize, usize), Fail> {
    engine.gpu.ctx.bind_to_thread()?;
    engine.stream().synchronize()?;
    Ok(engine.stream().context().mem_get_info()?)
}

fn record(engines: [&Engine; 2], phase: &str) -> Result<(), Fail> {
    for (card, engine) in engines.into_iter().enumerate() {
        let (free, total) = memory(engine)?;
        println!("memory\t{phase}\t{card}\t{free}\t{total}");
    }
    Ok(())
}

fn run() -> Result<(), Fail> {
    let mut args = std::env::args().skip(1);
    let source_dir = args
        .next()
        .ok_or("usage: mimo_resident_source_probe <source_dir>")?;
    if args.next().is_some() {
        return Err("usage: mimo_resident_source_probe <source_dir>".into());
    }
    let source = Arc::new(SafetensorsSource::open(Path::new(&source_dir))?);
    let (config, _, binding) = bind_pinned_text_source(&source)?;
    let cards = [Engine::new(0)?, Engine::new(1)?];
    let engines = [&cards[0], &cards[1]];
    if cards[0].stream().context().ordinal() == cards[1].stream().context().ordinal() {
        return Err("MiMo resident source probe needs two distinct GPU devices".into());
    }
    println!("type\tphase\tcard\tfree_bytes\ttotal_bytes");
    record(engines, "empty")?;

    let text = MiMoTextWeights::load(engines, source.clone())?;
    if text.layers.len() != 48 || text.routed.iter().filter(|row| row.is_some()).count() != 47 {
        return Err("MiMo text residency lost a trunk layer".into());
    }
    if text.embedding_row(42)?.len() != 4_096 {
        return Err("MiMo source embedding row is incomplete".into());
    }
    record(engines, "text")?;

    let vision = MiMoVisionWeights::load(&cards[0], source.as_ref(), &binding, &config)?;
    if vision.blocks.len() != 28 {
        return Err("MiMo vision residency lost a transformer block".into());
    }
    record(engines, "vision")?;

    let audio = MiMoAudioPatchWeights::load(&cards[1], source.as_ref(), &binding, &config)?;
    if audio.local_layers.len() != 6 || audio.speech_embeddings.len() != 20 {
        return Err("MiMo audio patch residency lost a table or local layer".into());
    }
    record(engines, "audio_patch")?;

    let mtp = Mtp3Weights::load(&cards[0], source)?;
    mtp.check_device(&cards[0])?;
    record(engines, "mtp3")?;
    std::hint::black_box((&text, &vision, &audio, &mtp));
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("mimo_resident_source_probe: {error}");
        std::process::exit(1);
    }
}
