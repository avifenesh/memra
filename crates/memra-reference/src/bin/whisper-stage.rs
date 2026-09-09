//! CPU-only stage runner. It consumes an explicit PCM fixture, never opens an audio device.
use memra_gguf::model_packs::whisper::PACK;
use memra_gguf::safetensors::StModel;
use memra_reference::ReferenceTensor;
use memra_reference::speech::encoder::{WhisperEncoder, WhisperNumeric};
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

fn write_f32(path: &Path, values: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path, bytes)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 4 && args[1] == "gelu" {
        let x = read_f32(Path::new(&args[2]))?;
        let y: Vec<_> = x
            .into_iter()
            .map(memra_reference::speech::encoder::gelu_erf)
            .collect();
        write_f32(Path::new(&args[3]), &y)?;
        return Ok(());
    }

    if args.len() < 5 || !["mel", "encoder", "norm"].contains(&args[1].as_str()) {
        return Err("usage: whisper-stage mel CHECKPOINT_DIR PCM.f32 OUTPUT.f32 | encoder CHECKPOINT_DIR MEL.f32 OUTPUT_DIR f32|f16".into());
    }
    let dir = Path::new(&args[2]);
    let plan = PACK.compile_plan(
        &std::fs::read_to_string(dir.join("config.json"))?,
        &std::fs::read_to_string(dir.join("preprocessor_config.json"))?,
    )?;
    let speech = plan.speech.as_ref().unwrap();
    if args[1] == "mel" {
        if args.len() != 5 {
            return Err("mel takes exactly three arguments".into());
        }
        let pcm = read_f32(Path::new(&args[3]))?;
        let output = WhisperFrontend::new(&speech.frontend)?.compute(&pcm)?;
        write_f32(Path::new(&args[4]), &output.data)?;
        println!(
            "stage=mel shape={:?} input_samples={} numeric=f32",
            output.shape,
            pcm.len()
        );
    } else {
        if args.len() != 6 {
            return Err("encoder requires an explicit numeric class".into());
        }
        let numeric = match args[5].as_str() {
            "f32" => WhisperNumeric::F32,
            "f16" => WhisperNumeric::F16,
            _ => return Err("numeric class must be f32 or f16".into()),
        };
        let source = StModel::open(dir)?;
        let started = std::time::Instant::now();
        let encoder = WhisperEncoder::load(speech, &source, numeric)?;
        println!(
            "loaded encoder numeric={numeric:?} elapsed={:.3}",
            started.elapsed().as_secs_f64()
        );
        if args[1] == "norm" {
            let x = read_f32(Path::new(&args[3]))?;
            let y = encoder.probe_first_norm(&x)?;
            write_f32(Path::new(&args[4]), &y)?;
            return Ok(());
        }
        let mel = ReferenceTensor::new(
            vec![
                speech.frontend.mel_bins as usize,
                speech.frontend.max_frames as usize,
            ],
            read_f32(Path::new(&args[3]))?,
        )?;
        let out = Path::new(&args[4]);
        std::fs::create_dir(out)?;
        encoder.encode_with_trace(&mel, |name, tensor| {
            write_f32(&out.join(format!("{name}.f32")), &tensor.data).map_err(|e| e.to_string())?;
            println!(
                "stage={name} shape={:?} numeric={numeric:?} elapsed={:.3}",
                tensor.shape,
                started.elapsed().as_secs_f64()
            );
            Ok(())
        })?;
    }
    Ok(())
}
