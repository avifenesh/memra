//! Speech token ids to a transcript, inside the engine.
//!
//! Until now the lane's text numbers came from an offline tokenizer, which is a fine way to
//! score a gate and not a thing a product can ship. This turns ids into text with the
//! checkpoint's own vocabulary through `memra_tokenizer::detokenize::Detokenizer`, which
//! decodes without claiming it can encode.
//!
//! Timestamps are dropped here rather than by the vocabulary. Whisper's export does not flag
//! them special, and a vocabulary that guessed from the shape of a piece would be guessing.
//! The plan knows where its timestamps start, so the caller does.

use memra_gguf::model_plan::speech::WhisperPlan;
use memra_tokenizer::detokenize::Detokenizer;

/// The text a run of Whisper ids says, with control markers and timestamps removed.
pub fn whisper_transcript(plan: &WhisperPlan, detokenizer: &Detokenizer, ids: &[u32]) -> String {
    let decode = &plan.decode;
    let text: Vec<u32> = ids
        .iter()
        .copied()
        .filter(|&id| id < decode.timestamp_begin && id != decode.eos_token)
        .collect();
    detokenizer.decode(&text)
}

/// The same ids with their timestamps kept, for an alignment view.
pub fn whisper_transcript_with_timestamps(
    plan: &WhisperPlan,
    detokenizer: &Detokenizer,
    ids: &[u32],
) -> String {
    let kept: Vec<u32> = ids
        .iter()
        .copied()
        .filter(|&id| id != plan.decode.eos_token)
        .collect();
    detokenizer.decode_with_markers(&kept)
}

#[cfg(test)]
mod tests;
