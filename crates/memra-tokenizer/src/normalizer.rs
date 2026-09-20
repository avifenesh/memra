//! Declared input normalization. The pinned native Unicode tables match HF tokenizers 0.22.2.
use crate::json;
use std::borrow::Cow;
use unicode_normalization_alignments::{IsNormalized, UnicodeNormalization, is_nfc_quick};

/// Unicode data used by the NFC program, part of its effective interpretation.
pub const NFC_UNICODE_VERSION: (u64, u64, u64) = unicode_normalization_alignments::UNICODE_VERSION;

/// Preserve the declared program, including supported Sequence wrappers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NormalizationProgram {
    Identity,
    Nfc,
    Sequence(Vec<NormalizationProgram>),
}

impl NormalizationProgram {
    pub(crate) fn parse(value: Option<&json::Value>) -> Result<Self, String> {
        match value {
            None | Some(json::Value::Null) => Ok(Self::Identity),
            Some(value) => Self::stage(value, 0, &mut 0),
        }
    }

    fn stage(value: &json::Value, depth: usize, count: &mut usize) -> Result<Self, String> {
        *count += 1;
        if depth >= 32 || *count > 256 {
            return Err("normalizer program exceeds 32 levels or 256 stages".into());
        }
        let kind = value
            .get("type")
            .and_then(json::Value::as_str)
            .ok_or("normalizer stage must be an object with a string type")?;
        match kind {
            "NFC" => Ok(Self::Nfc),
            "Sequence" => {
                let stages = value
                    .get("normalizers")
                    .and_then(json::Value::as_arr)
                    .ok_or("normalizer Sequence requires a normalizers array")?;
                stages
                    .iter()
                    .map(|stage| Self::stage(stage, depth + 1, count))
                    .collect::<Result<Vec<_>, _>>()
                    .map(Self::Sequence)
            }
            _ => Err(format!(
                "unsupported normalizer '{kind}': only NFC and NFC-only Sequence programs are supported"
            )),
        }
    }

    fn has_nfc(&self) -> bool {
        match self {
            Self::Identity => false,
            Self::Nfc => true,
            Self::Sequence(stages) => stages.iter().any(Self::has_nfc),
        }
    }

    pub(crate) fn apply<'a>(&self, text: &'a str) -> Cow<'a, str> {
        // NFC is idempotent: an NFC-only sequence has the same result as one NFC stage.
        // Keep the original sequence above for inspection instead of erasing its declaration.
        if !self.has_nfc() || is_nfc_quick(text.chars()) == IsNormalized::Yes {
            Cow::Borrowed(text)
        } else {
            Cow::Owned(text.nfc().map(|(ch, _alignment)| ch).collect())
        }
    }
}
