//! CPU-only stage runner. It consumes an explicit PCM fixture, never opens an audio device.
use memra_gguf::model_packs::whisper::PACK;
use memra_reference::speech::frontend::WhisperFrontend;
use std::path::Path;

fn read_f32(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    if !bytes.len().is_multiple_of(4) {
        return Err("f32 input byte length is not divisible by four".into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 || args[1] != "mel" {
        return Err("usage: whisper-stage mel CHECKPOINT_DIR PCM.f32 OUTPUT.f32".into());
    }
    let dir = Path::new(&args[2]);
    let plan = PACK.compile_plan(
        &std::fs::read_to_string(dir.join("config.json"))?,
        &std::fs::read_to_string(dir.join("preprocessor_config.json"))?,
    )?;
    let pcm = read_f32(Path::new(&args[3]))?;
    let output = WhisperFrontend::new(&plan.speech.as_ref().unwrap().frontend)?.compute(&pcm)?;
    let bytes: Vec<u8> = output.data.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(&args[4], bytes)?;
    println!(
        "stage=mel shape={:?} input_samples={} numeric=f32",
        output.shape,
        pcm.len()
    );
    Ok(())
}
