//! Decode-only vocabulary loading.
//!
//! `Tokenizer::from_hf_dir` refuses a checkpoint whose pre-tokenizer memra has not ported,
//! because an unexact encode is fluent and invisible and poisons every id downstream. That
//! refusal is about encoding, and it should stay.
//!
//! A speech model never encodes user text on its transcription path. Its decoder emits ids and
//! something has to turn them into bytes. `Detokenizer` does that and only that: it has no
//! `encode`, so a model whose pre-tokenizer is unported can produce text without any way for an
//! unexact encode to reach a golden. Whisper large-v3 is exactly that case: byte-level BPE with
//! a pre-tokenizer split memra does not have.

use crate::json;
use crate::unicode;
use std::collections::HashSet;

/// A byte-level BPE vocabulary, loaded for decoding.
#[derive(Debug)]
pub struct Detokenizer {
    id_to_token: Vec<String>,
    /// Ids whose piece is a marker, not text. `decode` drops them.
    special: HashSet<u32>,
}

impl Detokenizer {
    /// Load `tokenizer.json` from a Hugging Face checkpoint directory.
    ///
    /// Only byte-level BPE is accepted. An SPM vocabulary decodes by different rules and is
    /// refused rather than decoded wrongly.
    pub fn from_hf_dir(dir: &std::path::Path) -> Result<Self, String> {
        let path = dir.join("tokenizer.json");
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        Self::from_tokenizer_json(&text)
    }

    pub fn from_tokenizer_json(text: &str) -> Result<Self, String> {
        let tj = json::parse(text).map_err(|e| format!("tokenizer.json: {e}"))?;
        let model = tj.get("model").ok_or("tokenizer.json: missing model")?;
        if let Some(kind) = model.get("type").and_then(|v| v.as_str())
            && kind != "BPE"
        {
            return Err(format!(
                "tokenizer.json: model type '{kind}' is not BPE, and decoding it by BPE rules \
                 would be wrong"
            ));
        }
        let vocab = model
            .get("vocab")
            .and_then(|v| v.as_obj())
            .ok_or("tokenizer.json: missing model.vocab")?;

        let mut entries: Vec<(u32, String)> = Vec::with_capacity(vocab.len());
        let mut highest = 0u32;
        for (piece, id) in vocab.iter() {
            let id = id
                .as_u64()
                .and_then(|v| u32::try_from(v).ok())
                .ok_or("tokenizer.json: vocabulary id is not a u32")?;
            highest = highest.max(id);
            entries.push((id, piece.clone()));
        }

        let empty: Vec<json::Value> = Vec::new();
        let added = tj
            .get("added_tokens")
            .and_then(|v| v.as_arr())
            .unwrap_or(&empty);
        let mut special = HashSet::new();
        for entry in added {
            let id = entry
                .get("id")
                .and_then(|v| v.as_u64())
                .and_then(|v| u32::try_from(v).ok())
                .ok_or("tokenizer.json: added token id is not a u32")?;
            let content = entry
                .get("content")
                .and_then(|v| v.as_str())
                .ok_or("tokenizer.json: added token has no content")?;
            highest = highest.max(id);
            entries.push((id, content.to_owned()));
            // An added token that is not marked special is ordinary text that happens to be
            // added; only the marked ones are dropped by `decode`.
            if entry
                .get("special")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                special.insert(id);
            }
        }

        let mut id_to_token = vec![String::new(); highest as usize + 1];
        for (id, piece) in entries {
            id_to_token[id as usize] = piece;
        }
        Ok(Self {
            id_to_token,
            special,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    pub fn is_special(&self, id: u32) -> bool {
        self.special.contains(&id)
    }

    /// Ids to text, dropping marker tokens.
    pub fn decode(&self, ids: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids, false)).into_owned()
    }

    /// Ids to text, keeping marker tokens as their literal pieces.
    pub fn decode_with_markers(&self, ids: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids, true)).into_owned()
    }

    /// Ids to their exact byte stream. A streaming caller has to hold an incomplete UTF-8
    /// suffix across token boundaries rather than let it become a replacement character.
    pub fn decode_bytes(&self, ids: &[u32], markers: bool) -> Vec<u8> {
        let mut bytes = Vec::new();
        for &id in ids {
            let Some(piece) = self.id_to_token.get(id as usize) else {
                continue;
            };
            if piece.is_empty() {
                continue;
            }
            if self.special.contains(&id) {
                if markers {
                    bytes.extend_from_slice(piece.as_bytes());
                }
                continue;
            }
            for c in piece.chars() {
                match unicode::unicode_to_byte(c) {
                    Some(b) => bytes.push(b),
                    None => {
                        let mut buffer = [0u8; 4];
                        bytes.extend_from_slice(c.encode_utf8(&mut buffer).as_bytes());
                    }
                }
            }
        }
        bytes
    }
}

#[cfg(test)]
mod tests;
