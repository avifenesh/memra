use super::*;

const TINY: &str = r#"{
  "model": {
    "type": "BPE",
    "vocab": {"Ġhello": 1, "Ġworld": 2, "!": 3, "Ã©": 4}
  },
  "added_tokens": [
    {"id": 5, "content": "<|startoftranscript|>", "special": true},
    {"id": 6, "content": "<|0.00|>", "special": true},
    {"id": 7, "content": "[literal]", "special": false}
  ]
}"#;

#[test]
fn byte_level_pieces_decode_to_their_bytes() {
    let d = Detokenizer::from_tokenizer_json(TINY).unwrap();
    assert_eq!(d.decode(&[1, 2, 3]), " hello world!");
    // Two byte-level chars, 0xC3 and 0xA9, are one UTF-8 character once decoded.
    assert_eq!(d.decode(&[4]), "é");
}

#[test]
fn markers_are_dropped_unless_they_are_asked_for() {
    let d = Detokenizer::from_tokenizer_json(TINY).unwrap();
    assert!(d.is_special(5));
    assert!(d.is_special(6));
    assert_eq!(d.decode(&[5, 6, 1, 3]), " hello!");
    assert_eq!(
        d.decode_with_markers(&[5, 6, 1, 3]),
        "<|startoftranscript|><|0.00|> hello!"
    );
}

#[test]
fn an_added_token_that_is_not_marked_special_is_text() {
    let d = Detokenizer::from_tokenizer_json(TINY).unwrap();
    assert!(!d.is_special(7));
    assert_eq!(d.decode(&[7]), "[literal]");
}

#[test]
fn an_unknown_id_is_skipped_rather_than_guessed() {
    let d = Detokenizer::from_tokenizer_json(TINY).unwrap();
    assert_eq!(d.decode(&[1, 9999, 3]), " hello!");
}

#[test]
fn a_non_bpe_vocabulary_is_refused() {
    let spm = r#"{"model": {"type": "Unigram", "vocab": {"a": 0}}}"#;
    let error = Detokenizer::from_tokenizer_json(spm).unwrap_err();
    assert!(error.contains("not BPE"), "{error}");
}

#[test]
fn the_real_whisper_vocabulary_decodes_hebrew_when_the_checkpoint_is_present() {
    let dir = std::path::Path::new(
        "/home/avifenesh/hebrew-asr-data/models/whisper-large-v3-ivrit-766847c9",
    );
    if !dir.exists() {
        eprintln!("skipping: whisper checkpoint is not on this machine");
        return;
    }
    let d = Detokenizer::from_hf_dir(dir).unwrap();
    assert_eq!(d.vocab_size(), 51866);
    // The opening of d1-000 window 0, straight out of the pinned oracle's beam-1 tokens.
    let ids = [50413u32, 6044, 97, 2986, 8573, 8817, 29526, 17333];
    // Whisper's timestamps are added tokens that its export does NOT flag special, so this
    // type does not drop them. The caller knows where its timestamps start and filters; the
    // vocabulary is not the place to guess that a piece shaped like a marker is one.
    assert!(!d.is_special(50413));
    assert!(d.decode(&ids).starts_with("<|0.96|>"));
    let text: Vec<u32> = ids.into_iter().filter(|&id| id < 50257).collect();
    assert_eq!(d.decode(&text), " בפרק של היום");
    // The transcript-start marker is flagged, and is dropped.
    assert!(d.is_special(50258));
    assert_eq!(d.decode(&[50258]), "");
}
