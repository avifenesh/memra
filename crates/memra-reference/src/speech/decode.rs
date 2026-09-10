//! Native Whisper beam-1 decode policy and the clip window program.
//!
//! Every constant this file needs comes from the compiled `WhisperPlan`, and every rule is
//! pinned against the offline rental oracle. No external decoder, tokenizer or feature
//! extractor runs here. This is a reference operation, not a serving route: it produces token
//! ids, and turning ids back into text is still an unbuilt native surface.

use super::decoder::WhisperDecoder;
use super::encoder::WhisperEncoder;
use super::frontend::WhisperFrontend;
use crate::ReferenceTensor;
use memra_gguf::model_plan::speech::WhisperPlan;

/// One decode window is a fixed 30-second mel field, whatever the audio behind it.
pub const WINDOW_FRAMES: usize = 3000;
/// Two mel frames per timestamp unit: hop 160 at 16 kHz gives 0.01 s per frame, 0.02 s per unit.
pub const FRAMES_PER_TIMESTAMP_UNIT: usize = 2;

#[derive(Debug, Clone)]
pub struct WhisperWindow {
    pub index: usize,
    pub seek_frame: usize,
    pub segment_frames: usize,
    /// Emitted ids, including the terminating EOS when the window terminated on its own.
    pub tokens: Vec<u32>,
    /// True when the window stopped on the generated-token cap instead of EOS.
    pub reached_cap: bool,
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut value = f32::NEG_INFINITY;
    for (i, &x) in logits.iter().enumerate() {
        if x > value {
            best = i;
            value = x;
        }
    }
    best as u32
}

fn log_sum_exp(values: &[f32]) -> f32 {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return f32::NEG_INFINITY;
    }
    let mut sum = 0.0f32;
    for &x in values {
        if x.is_finite() {
            sum += (x - max).exp();
        }
    }
    max + sum.ln()
}

/// Mask the ids the deterministic transcription program forbids at this step.
///
/// `generated` holds only the tokens this window emitted; the language/task prefix is not part
/// of it. Order matters: static suppression, then the first-step blank rule, then the timestamp
/// grammar, then the rule that forces a timestamp when the timestamps outweigh every text token.
pub fn apply_decode_policy(
    plan: &WhisperPlan,
    logits: &mut [f32],
    generated: &[u32],
) -> Result<(), String> {
    let decode = &plan.decode;
    let vocab = plan.vocab_size as usize;
    if logits.len() != vocab {
        return Err("decode policy needs one logit per vocabulary id".into());
    }
    let timestamp_begin = decode.timestamp_begin as usize;
    let eos = decode.eos_token as usize;
    if timestamp_begin >= vocab || eos >= timestamp_begin {
        return Err("unsupported Whisper decode token geometry".into());
    }
    for &id in decode.suppress_tokens {
        let id = id as usize;
        if id >= timestamp_begin {
            return Err("suppressed id is a timestamp".into());
        }
        logits[id] = f32::NEG_INFINITY;
    }
    if generated.is_empty() {
        logits[decode.blank_token as usize] = f32::NEG_INFINITY;
        logits[eos] = f32::NEG_INFINITY;
    }
    let is_timestamp = |t: u32| t >= decode.timestamp_begin;
    let last_was_timestamp = generated.last().copied().is_some_and(is_timestamp);
    // A missing penultimate token counts as a timestamp: the window opens on one.
    let penultimate_was_timestamp =
        generated.len() < 2 || is_timestamp(generated[generated.len() - 2]);
    if last_was_timestamp {
        if penultimate_was_timestamp {
            // Two timestamps closed a segment, so the next id must be text.
            logits[timestamp_begin..].fill(f32::NEG_INFINITY);
        } else {
            // One timestamp opened a segment, so the next id must close it or end the window.
            logits[..eos].fill(f32::NEG_INFINITY);
        }
    }
    if let Some(&last) = generated.iter().rev().find(|&&t| is_timestamp(t)) {
        // Time never runs backwards inside a window. An open segment may repeat its own
        // timestamp; a closed one may not.
        let limit = if last_was_timestamp && !penultimate_was_timestamp {
            last as usize
        } else {
            last as usize + 1
        };
        logits[timestamp_begin..limit.min(vocab)].fill(f32::NEG_INFINITY);
    }
    if generated.is_empty() {
        let last_allowed = timestamp_begin + decode.max_initial_timestamp_index as usize;
        if last_allowed + 1 < vocab {
            logits[last_allowed + 1..].fill(f32::NEG_INFINITY);
        }
    }
    // Force a timestamp when the timestamps hold more probability than any single text id.
    // The softmax normalizer is the same on both sides of that comparison, so it cancels and
    // the raw logits carry the decision.
    let timestamp_mass = log_sum_exp(&logits[timestamp_begin..]);
    let best_text = logits[..timestamp_begin]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    if timestamp_mass > best_text {
        logits[..timestamp_begin].fill(f32::NEG_INFINITY);
    }
    if !logits.iter().any(|x| x.is_finite()) {
        return Err("Whisper decode policy left no admissible token".into());
    }
    Ok(())
}

/// Beam-1 decode of one encoded window. The prefix is the plan's start/language/task ids;
/// no previous-window text conditions this window.
pub fn decode_window(
    decoder: &WhisperDecoder,
    plan: &WhisperPlan,
    encoded: &ReferenceTensor,
) -> Result<(Vec<u32>, bool), String> {
    let decode = &plan.decode;
    if decode.beam_size != 1 || !decode.deterministic {
        return Err("this reference decode is beam-1 deterministic only".into());
    }
    let mut session = decoder.start(encoded)?;
    let prefix = [decode.start_token, decode.language_token, decode.task_token];
    let mut logits = Vec::new();
    for &token in &prefix {
        logits = session.step(token)?.logits;
    }
    let cap = decode.max_generated_tokens as usize;
    let mut generated: Vec<u32> = Vec::new();
    loop {
        apply_decode_policy(plan, &mut logits, &generated)?;
        let next = argmax(&logits);
        generated.push(next);
        if next == decode.eos_token {
            return Ok((generated, false));
        }
        if generated.len() >= cap {
            return Ok((generated, true));
        }
        logits = session.step(next)?.logits;
    }
}

/// Mel frames the clip program can seek over: one 30-second field of trailing pad is analysis
/// context, not audio, so it is never a window start.
pub fn clip_content_frames(mel: &ReferenceTensor) -> Result<usize, String> {
    if mel.shape.len() != 2 || mel.shape[1] == 0 {
        return Err("clip log-mel must be [mel_bins, frames]".into());
    }
    Ok(mel.shape[1] - 1)
}

/// Cut one window out of the clip log-mel. Frames past the audio are zero, which is what the
/// pinned extractor writes there, not the analytic silence floor a padded single-window run has.
pub fn window_mel(mel: &ReferenceTensor, seek: usize, segment: usize) -> Result<Vec<f32>, String> {
    let bins = mel.shape[0];
    let frames = mel.shape[1];
    if segment == 0 || segment > WINDOW_FRAMES || seek + segment > frames {
        return Err("invalid Whisper window extent".into());
    }
    let mut window = vec![0.0f32; bins * WINDOW_FRAMES];
    for bin in 0..bins {
        window[bin * WINDOW_FRAMES..bin * WINDOW_FRAMES + segment]
            .copy_from_slice(&mel.data[bin * frames + seek..bin * frames + seek + segment]);
    }
    Ok(window)
}

/// How far the clip program advances after a window, from that window's own tokens.
///
/// A window that ends on a closed segment resumes at that segment's end. A window that ends
/// with one open timestamp, or with no timestamp pair at all, consumes the whole field.
pub fn seek_advance(plan: &WhisperPlan, tokens: &[u32], segment_frames: usize) -> usize {
    let begin = plan.decode.timestamp_begin;
    let eos = plan.decode.eos_token;
    let body: &[u32] = match tokens.split_last() {
        Some((&last, head)) if last == eos => head,
        _ => tokens,
    };
    let is_timestamp = |t: u32| t >= begin;
    let single_timestamp_ending = body.len() >= 2
        && is_timestamp(body[body.len() - 1])
        && !is_timestamp(body[body.len() - 2]);
    let last_pair = (1..body.len())
        .rev()
        .find(|&i| is_timestamp(body[i]) && is_timestamp(body[i - 1]));
    match last_pair {
        Some(i) if !single_timestamp_ending => {
            (body[i - 1] - begin) as usize * FRAMES_PER_TIMESTAMP_UNIT
        }
        _ => segment_frames,
    }
}

/// Run the whole clip: native log-mel, then window by window native encode and beam-1 decode.
/// `observe` sees each finished window so a long sweep can bank progress as it goes.
pub fn transcribe_clip(
    frontend: &WhisperFrontend,
    encoder: &WhisperEncoder,
    decoder: &WhisperDecoder,
    plan: &WhisperPlan,
    pcm: &[f32],
    mut observe: impl FnMut(&WhisperWindow) -> Result<(), String>,
) -> Result<Vec<WhisperWindow>, String> {
    let mel = frontend.compute_clip(pcm)?;
    let content = clip_content_frames(&mel)?;
    let bins = plan.frontend.mel_bins as usize;
    if mel.shape[0] != bins {
        return Err("clip log-mel channel count disagrees with the plan".into());
    }
    let mut windows = Vec::new();
    let mut seek = 0usize;
    while seek < content {
        let segment = WINDOW_FRAMES.min(content - seek);
        let field = window_mel(&mel, seek, segment)?;
        let field =
            ReferenceTensor::new(vec![bins, WINDOW_FRAMES], field).map_err(|e| e.to_string())?;
        let encoded = encoder.encode(&field)?;
        let (tokens, reached_cap) = decode_window(decoder, plan, &encoded)?;
        let advance = seek_advance(plan, &tokens, segment);
        if advance == 0 {
            return Err("clip window program did not advance".into());
        }
        let window = WhisperWindow {
            index: windows.len(),
            seek_frame: seek,
            segment_frames: segment,
            tokens,
            reached_cap,
        };
        observe(&window)?;
        windows.push(window);
        seek += advance;
    }
    Ok(windows)
}

#[cfg(test)]
mod tests;
