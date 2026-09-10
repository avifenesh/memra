//! Turn a banked run of Whisper token ids into text with the checkpoint's own vocabulary.
//!
//! usage: speech-text CHECKPOINT_DIR CLIP_DIR
//!
//! Reads `window-NNN-tokens.i32.bin` in order and writes `transcript.txt` and
//! `transcript-timestamps.txt` beside them. No audio, no model execution.
use memra_gguf::model_packs::whisper::PACK;
use memra_reference::speech::text::{whisper_transcript, whisper_transcript_with_timestamps};
use memra_tokenizer::detokenize::Detokenizer;
use std::path::{Path, PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: speech-text CHECKPOINT_DIR CLIP_DIR".into());
    }
    let checkpoint = PathBuf::from(&args[1]);
    let clip = PathBuf::from(&args[2]);
    let plan = PACK.compile_plan(
        &std::fs::read_to_string(checkpoint.join("config.json"))?,
        &std::fs::read_to_string(checkpoint.join("preprocessor_config.json"))?,
    )?;
    let speech = plan
        .speech
        .as_ref()
        .ok_or("compiled plan carries no speech")?;
    let detokenizer = Detokenizer::from_hf_dir(Path::new(&checkpoint))?;

    let mut ids: Vec<u32> = Vec::new();
    let mut index = 0usize;
    loop {
        let path = clip.join(format!("window-{index:03}-tokens.i32.bin"));
        if !path.exists() {
            break;
        }
        let bytes = std::fs::read(&path)?;
        for chunk in bytes.chunks_exact(4) {
            ids.push(i32::from_le_bytes(chunk.try_into().unwrap()) as u32);
        }
        index += 1;
    }
    if index == 0 {
        return Err("clip directory holds no window token files".into());
    }
    let transcript = whisper_transcript(speech, &detokenizer, &ids);
    std::fs::write(clip.join("transcript.txt"), &transcript)?;
    std::fs::write(
        clip.join("transcript-timestamps.txt"),
        whisper_transcript_with_timestamps(speech, &detokenizer, &ids),
    )?;
    println!(
        "windows={index} tokens={} characters={}",
        ids.len(),
        transcript.chars().count()
    );
    Ok(())
}
