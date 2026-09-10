//! CPU-only stage runner. It consumes an explicit PCM fixture, never opens an audio device.
use memra_gguf::model_packs::whisper::PACK;
use memra_gguf::safetensors::StModel;
use memra_reference::ReferenceTensor;
use memra_reference::speech::decode::{WhisperWindow, transcribe_clip};
use memra_reference::speech::decoder::WhisperDecoder;
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

    if args.len() < 5
        || ![
            "mel",
            "mel-clip",
            "encoder",
            "norm",
            "decoder",
            "transcribe",
        ]
        .contains(&args[1].as_str())
    {
        return Err("usage: whisper-stage mel CHECKPOINT_DIR PCM.f32 OUTPUT.f32 | encoder CHECKPOINT_DIR MEL.f32 OUTPUT_DIR f32|f16".into());
    }
    let dir = Path::new(&args[2]);
    let plan = PACK.compile_plan(
        &std::fs::read_to_string(dir.join("config.json"))?,
        &std::fs::read_to_string(dir.join("preprocessor_config.json"))?,
    )?;
    let speech = plan.speech.as_ref().unwrap();
    if args[1] == "decoder" {
        if args.len() != 7 {
            return Err("usage: whisper-stage decoder CHECKPOINT_DIR ENCODER.f32 TOKENS.txt OUTPUT_DIR f32|f16".into());
        }
        let numeric = match args[6].as_str() {
            "f32" => WhisperNumeric::F32,
            "f16" => WhisperNumeric::F16,
            _ => return Err("numeric class must be f32 or f16".into()),
        };
        let encoded = read_f32(Path::new(&args[3]))?;
        let h = speech.hidden_size as usize;
        if encoded.is_empty() || !encoded.len().is_multiple_of(h) {
            return Err("invalid encoder fixture extent".into());
        }
        let encoded = ReferenceTensor::new(vec![encoded.len() / h, h], encoded)?;
        let token_text = std::fs::read_to_string(&args[4])?;
        let tokens = token_text
            .split_whitespace()
            .map(str::parse::<u32>)
            .collect::<Result<Vec<_>, _>>()?;
        if tokens.is_empty() || tokens.len() > speech.target_positions as usize {
            return Err("invalid decoder token fixture length".into());
        }
        let source = StModel::open(dir)?;
        let decoder = WhisperDecoder::load(speech, &source, numeric)?;
        let started = std::time::Instant::now();
        let mut session = decoder.start(&encoded)?;
        let out = Path::new(&args[5]);
        std::fs::create_dir(out)?;
        std::fs::write(out.join("input-tokens.txt"), token_text)?;
        std::fs::write(out.join("numeric.txt"), format!("{numeric:?}\n"))?;
        write_f32(&out.join("encoder-input.f32"), &encoded.data)?;
        let mut summary = String::from("position\tinput_token\traw_argmax\n");
        for token in tokens {
            let result = session.step(token)?;
            write_f32(
                &out.join(format!("step-{:02}-hidden.f32", result.position)),
                &result.hidden,
            )?;
            write_f32(
                &out.join(format!("step-{:02}-logits.f32", result.position)),
                &result.logits,
            )?;
            let best = result
                .logits
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |best, (i, &x)| {
                    if x > best.1 { (i, x) } else { best }
                })
                .0;
            summary.push_str(&format!("{}\t{token}\t{best}\n", result.position));
            println!(
                "decoder step={} input={token} raw_argmax={best} numeric={numeric:?} elapsed={:.3}",
                result.position,
                started.elapsed().as_secs_f64()
            );
        }
        std::fs::write(out.join("steps.tsv"), summary)?;
        return Ok(());
    }
    if args[1] == "transcribe" {
        if args.len() != 6 {
            return Err(
                "usage: whisper-stage transcribe CHECKPOINT_DIR PCM.f32 OUTPUT_DIR f32|f16".into(),
            );
        }
        let numeric = match args[5].as_str() {
            "f32" => WhisperNumeric::F32,
            "f16" => WhisperNumeric::F16,
            _ => return Err("numeric class must be f32 or f16".into()),
        };
        let pcm = read_f32(Path::new(&args[3]))?;
        let out = Path::new(&args[4]);
        std::fs::create_dir(out)?;
        let source = StModel::open(dir)?;
        let started = std::time::Instant::now();
        let frontend = WhisperFrontend::new(&speech.frontend)?;
        let encoder = WhisperEncoder::load(speech, &source, numeric)?;
        let decoder = WhisperDecoder::load(speech, &source, numeric)?;
        println!(
            "loaded transcribe numeric={numeric:?} samples={} elapsed={:.3}",
            pcm.len(),
            started.elapsed().as_secs_f64()
        );
        // Bank each window as it finishes: a sweep that is interrupted keeps what it proved.
        let bank = |window: &WhisperWindow| -> Result<(), String> {
            let bytes: Vec<u8> = window
                .tokens
                .iter()
                .flat_map(|t| (*t as i32).to_le_bytes())
                .collect();
            std::fs::write(
                out.join(format!("window-{:03}-tokens.i32.bin", window.index)),
                bytes,
            )
            .map_err(|e| e.to_string())?;
            println!(
                "window={} seek={} segment={} tokens={} capped={} elapsed={:.3}",
                window.index,
                window.seek_frame,
                window.segment_frames,
                window.tokens.len(),
                window.reached_cap,
                started.elapsed().as_secs_f64()
            );
            Ok(())
        };
        let windows = transcribe_clip(&frontend, &encoder, &decoder, speech, &pcm, bank)?;
        let mut summary = String::from("index\tseek_frame\tsegment_frames\ttokens\treached_cap\n");
        for w in &windows {
            summary.push_str(&format!(
                "{}\t{}\t{}\t{}\t{}\n",
                w.index,
                w.seek_frame,
                w.segment_frames,
                w.tokens.len(),
                w.reached_cap
            ));
        }
        std::fs::write(out.join("windows.tsv"), summary)?;
        std::fs::write(out.join("numeric.txt"), format!("{numeric:?}\n"))?;
        println!(
            "transcribe windows={} elapsed={:.3}",
            windows.len(),
            started.elapsed().as_secs_f64()
        );
        return Ok(());
    }
    if args[1] == "mel" || args[1] == "mel-clip" {
        if args.len() != 5 {
            return Err("mel takes exactly three arguments".into());
        }
        let pcm = read_f32(Path::new(&args[3]))?;
        let frontend = WhisperFrontend::new(&speech.frontend)?;
        let output = if args[1] == "mel-clip" {
            frontend.compute_clip(&pcm)?
        } else {
            frontend.compute(&pcm)?
        };
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
        // Bank the mel this run actually consumed. Without it a capture cannot say whether it
        // measures the encoder alone (reference log-mel in) or the frontend and encoder stacked.
        write_f32(&out.join("input-mel.f32"), &mel.data)?;
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
