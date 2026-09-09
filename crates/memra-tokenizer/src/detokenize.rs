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

/// A SentencePiece vocabulary, loaded for decoding.
///
/// The pieces come out of the model's own serialized `ModelProto`, read with a minimal
/// length-delimited reader: field 1 is the repeated piece record, and inside it field 1 is the
/// piece text and field 3 is its type. Nothing else in the proto is interpreted, and an
/// unrecognized wire type is an error rather than a skip, so a different proto cannot be read
/// as this one.
#[derive(Debug)]
pub struct SpmDetokenizer {
    pieces: Vec<String>,
    kinds: Vec<SpmPieceKind>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpmPieceKind {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Byte,
    Unused,
}

/// SentencePiece writes a word boundary as this character, not as a space.
pub const WORD_BOUNDARY: char = '\u{2581}';

impl SpmDetokenizer {
    /// Parse a serialized SentencePiece `ModelProto`.
    pub fn from_proto(bytes: &[u8]) -> Result<Self, String> {
        let mut pieces = Vec::new();
        let mut kinds = Vec::new();
        let mut cursor = 0usize;
        while cursor < bytes.len() {
            let (key, next) = read_varint(bytes, cursor)?;
            cursor = next;
            let field = key >> 3;
            let wire = key & 7;
            match (field, wire) {
                (1, 2) => {
                    let (len, next) = read_varint(bytes, cursor)?;
                    // The record starts after its length varint, not before it.
                    let end = next
                        .checked_add(len as usize)
                        .filter(|&e| e <= bytes.len())
                        .ok_or("sentencepiece proto: piece record runs past the model")?;
                    let (piece, kind) = read_piece(&bytes[next..end])?;
                    pieces.push(piece);
                    kinds.push(kind);
                    cursor = end;
                }
                (_, wire) => cursor = skip_field(bytes, cursor, wire)?,
            }
        }
        if pieces.is_empty() {
            return Err("sentencepiece proto: no pieces".into());
        }
        Ok(Self { pieces, kinds })
    }

    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    pub fn piece(&self, id: u32) -> Option<&str> {
        self.pieces.get(id as usize).map(String::as_str)
    }

    pub fn kind(&self, id: u32) -> Option<SpmPieceKind> {
        self.kinds.get(id as usize).copied()
    }

    /// Ids to text. Control and unknown pieces are dropped, byte pieces become their byte, the
    /// word-boundary character becomes a space, and the one leading space the boundary
    /// convention produces is removed.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            let Some(piece) = self.pieces.get(id as usize) else {
                continue;
            };
            match self.kinds[id as usize] {
                SpmPieceKind::Control | SpmPieceKind::Unknown | SpmPieceKind::Unused => continue,
                SpmPieceKind::Byte => {
                    if let Some(value) = piece
                        .strip_prefix("<0x")
                        .and_then(|rest| rest.strip_suffix('>'))
                        .and_then(|hex| u8::from_str_radix(hex, 16).ok())
                    {
                        bytes.push(value);
                        continue;
                    }
                    bytes.extend_from_slice(piece.as_bytes());
                }
                SpmPieceKind::Normal | SpmPieceKind::UserDefined => {
                    for c in piece.chars() {
                        if c == WORD_BOUNDARY {
                            bytes.push(b' ');
                        } else {
                            let mut buffer = [0u8; 4];
                            bytes.extend_from_slice(c.encode_utf8(&mut buffer).as_bytes());
                        }
                    }
                }
            }
        }
        let text = String::from_utf8_lossy(&bytes).into_owned();
        text.strip_prefix(' ').map(str::to_owned).unwrap_or(text)
    }
}

fn read_varint(bytes: &[u8], mut cursor: usize) -> Result<(u64, usize), String> {
    let mut value = 0u64;
    let mut shift = 0u32;
    loop {
        let byte = *bytes
            .get(cursor)
            .ok_or("sentencepiece proto: truncated varint")?;
        cursor += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok((value, cursor));
        }
        shift += 7;
        if shift > 63 {
            return Err("sentencepiece proto: varint is too long".into());
        }
    }
}

fn skip_field(bytes: &[u8], cursor: usize, wire: u64) -> Result<usize, String> {
    match wire {
        0 => Ok(read_varint(bytes, cursor)?.1),
        1 => cursor
            .checked_add(8)
            .filter(|&e| e <= bytes.len())
            .ok_or_else(|| "sentencepiece proto: truncated fixed64".to_owned()),
        2 => {
            let (len, next) = read_varint(bytes, cursor)?;
            next.checked_add(len as usize)
                .filter(|&e| e <= bytes.len())
                .ok_or_else(|| "sentencepiece proto: truncated length-delimited field".to_owned())
        }
        5 => cursor
            .checked_add(4)
            .filter(|&e| e <= bytes.len())
            .ok_or_else(|| "sentencepiece proto: truncated fixed32".to_owned()),
        other => Err(format!(
            "sentencepiece proto: unsupported wire type {other}"
        )),
    }
}

fn read_piece(bytes: &[u8]) -> Result<(String, SpmPieceKind), String> {
    let mut piece = String::new();
    let mut kind = SpmPieceKind::Normal;
    let mut cursor = 0usize;
    while cursor < bytes.len() {
        let (key, next) = read_varint(bytes, cursor)?;
        cursor = next;
        match (key >> 3, key & 7) {
            (1, 2) => {
                let (len, next) = read_varint(bytes, cursor)?;
                let end = next
                    .checked_add(len as usize)
                    .filter(|&e| e <= bytes.len())
                    .ok_or("sentencepiece proto: piece text runs past its record")?;
                piece = std::str::from_utf8(&bytes[next..end])
                    .map_err(|_| "sentencepiece proto: piece text is not utf-8")?
                    .to_owned();
                cursor = end;
            }
            (3, 0) => {
                let (value, next) = read_varint(bytes, cursor)?;
                kind = match value {
                    1 => SpmPieceKind::Normal,
                    2 => SpmPieceKind::Unknown,
                    3 => SpmPieceKind::Control,
                    4 => SpmPieceKind::UserDefined,
                    // The enum is NORMAL 1, UNKNOWN 2, CONTROL 3, USER_DEFINED 4, UNUSED 5,
                    // BYTE 6. Swapping the last two would turn byte fallbacks into dropped
                    // pieces and unused slots into raw bytes.
                    5 => SpmPieceKind::Unused,
                    6 => SpmPieceKind::Byte,
                    other => {
                        return Err(format!("sentencepiece proto: unknown piece type {other}"));
                    }
                };
                cursor = next;
            }
            (_, wire) => cursor = skip_field(bytes, cursor, wire)?,
        }
    }
    Ok((piece, kind))
}
