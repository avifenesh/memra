//! HF added-vocabulary extraction on either side of the declared normalizer.
use crate::{Fragment, json, normalizer::NormalizationProgram};
use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};

#[derive(Clone, Debug)]
pub(crate) struct AddedToken {
    pub(crate) id: u32,
    pub(crate) content: String,
    pub(crate) special: bool,
    normalized: bool,
    single_word: bool,
    lstrip: bool,
    rstrip: bool,
}

impl AddedToken {
    pub(crate) fn parse(value: &json::Value) -> Result<Self, String> {
        fn flag(value: &json::Value, name: &str, default: bool) -> Result<bool, String> {
            match value.get(name) {
                None => Ok(default),
                Some(json::Value::Bool(value)) => Ok(*value),
                Some(_) => Err(format!("added_tokens.{name} must be boolean")),
            }
        }
        let id = value
            .get("id")
            .and_then(json::Value::as_u64)
            .and_then(|id| u32::try_from(id).ok())
            .ok_or("added_tokens entry requires a u32 id")?;
        let content = value
            .get("content")
            .and_then(json::Value::as_str)
            .ok_or("added_tokens entry requires string content")?
            .to_owned();
        let special = flag(value, "special", false)?;
        Ok(Self {
            id,
            content,
            special,
            // Retain the existing concise-fixture convention, equivalent to AddedToken::from.
            // Full HF serialization always carries an explicit normalized flag.
            normalized: flag(value, "normalized", !special)?,
            single_word: flag(value, "single_word", false)?,
            lstrip: flag(value, "lstrip", false)?,
            rstrip: flag(value, "rstrip", false)?,
        })
    }
}

struct MatchStage {
    matcher: AhoCorasick,
    tokens: Vec<AddedToken>,
    strips_whitespace: bool,
}

impl MatchStage {
    fn new(tokens: Vec<AddedToken>, normalizer: &NormalizationProgram) -> Result<Self, String> {
        let patterns: Vec<_> = tokens
            .iter()
            .map(|token| {
                if token.normalized {
                    normalizer.apply(&token.content).into_owned()
                } else {
                    token.content.clone()
                }
            })
            .collect();
        let matcher = AhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostLongest)
            .build(&patterns)
            .map_err(|error| format!("cannot build added-token matcher: {error}"))?;
        let strips_whitespace = tokens.iter().any(|token| token.lstrip || token.rstrip);
        Ok(Self {
            matcher,
            tokens,
            strips_whitespace,
        })
    }

    fn partition(&self, text: &str, parse_special: bool) -> Vec<Fragment> {
        let mut result = Vec::new();
        let mut cursor = 0;
        // A whitespace token with rstrip can match many times inside one whitespace run.
        // Re-scanning that entire suffix for each match would make input processing quadratic.
        let mut whitespace = Vec::new();
        if self.strips_whitespace {
            let mut start = None;
            for (offset, ch) in text.char_indices() {
                if ch.is_whitespace() {
                    start.get_or_insert(offset);
                } else if let Some(start) = start.take() {
                    whitespace.push((start, offset));
                }
            }
            if let Some(start) = start {
                whitespace.push((start, text.len()));
            }
        }
        // HF filters a selected match after leftmost-longest matching; removing special
        // patterns beforehand would expose shorter overlapping ordinary-token matches.
        for found in self.matcher.find_iter(text) {
            let token = &self.tokens[found.pattern().as_usize()];
            if token.special && !parse_special {
                continue;
            }
            let mut start = found.start();
            let mut end = found.end();
            if token.single_word
                && (text[..start]
                    .chars()
                    .next_back()
                    .is_some_and(regex_syntax::is_word_character)
                    || text[end..]
                        .chars()
                        .next()
                        .is_some_and(regex_syntax::is_word_character))
            {
                continue;
            }
            if token.lstrip {
                let run = whitespace.partition_point(|&(_, end)| end < start);
                if let Some(&(left, right)) = whitespace.get(run)
                    && left < start
                    && start <= right
                {
                    start = left;
                }
                start = start.max(cursor);
            }
            if token.rstrip {
                let run = whitespace.partition_point(|&(_, right)| right < end);
                if let Some(&(left, right)) = whitespace.get(run)
                    && left <= end
                    && end < right
                {
                    end = right;
                }
            }
            // HF discards a selected match when whitespace expansion consumed its
            // complete adjusted span. Keep nonempty overlapping matches unchanged.
            if start == end {
                continue;
            }
            if cursor < start {
                result.push(Fragment::Text(text[cursor..start].to_owned()));
            }
            result.push(Fragment::Token(token.id));
            cursor = end;
        }
        if cursor < text.len() {
            result.push(Fragment::Text(text[cursor..].to_owned()));
        }
        result
    }
}

pub(crate) struct HfInput {
    raw: MatchStage,
    normalized: MatchStage,
}

impl HfInput {
    pub(crate) fn new(
        mut tokens: Vec<AddedToken>,
        normalizer: &NormalizationProgram,
    ) -> Result<Self, String> {
        tokens.retain(|token| !token.content.is_empty());
        // HF rebuilds its matchers with special tokens first, then ordinary additions.
        // Preserve declaration order within each class, including noncanonical files:
        // that order breaks ties when distinct spellings normalize to the same content.
        tokens.sort_by_key(|token| !token.special);
        let (normalized, raw) = tokens.into_iter().partition(|token| token.normalized);
        Ok(Self {
            raw: MatchStage::new(raw, normalizer)?,
            normalized: MatchStage::new(normalized, normalizer)?,
        })
    }

    pub(crate) fn partition(
        &self,
        text: &str,
        normalizer: &NormalizationProgram,
        parse_special: bool,
    ) -> Vec<Fragment> {
        let mut result = Vec::new();
        for fragment in self.raw.partition(text, parse_special) {
            match fragment {
                Fragment::Token(id) => result.push(Fragment::Token(id)),
                Fragment::Text(text) => {
                    let normalized = normalizer.apply(&text);
                    result.extend(self.normalized.partition(&normalized, parse_special));
                }
            }
        }
        result
    }
}
