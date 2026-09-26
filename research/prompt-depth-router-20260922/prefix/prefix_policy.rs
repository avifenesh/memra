//! Only a bounded prefix is passed to decoding and lexical classification.
use super::router::{Decision, Kind, Profile, select_depth};
use std::time::Instant;

pub const MAX_PREFIX_TOKENS: usize = 256;
pub const MAX_PREFIX_BYTES: usize = 16 * 1024;
const PROFILE: Profile = Profile {
    prose: 2,
    code: 4,
    numeric: 4,
    fallback: 3,
};

#[derive(Debug)]
pub struct Selection {
    pub decision: Decision,
    pub token_count: usize,
    pub decoded_bytes: usize,
    pub inspected_bytes: usize,
    pub elapsed_ns: u64,
    pub prefix: Vec<u8>,
}

/// `user_ids` excludes the template suffix. `leading_skip` handles a tokenizer
/// piece crossing the template/user byte boundary; it does not consume a token.
/// The decoder is called exactly once, with at most `budget` user tokens.
pub fn select<F>(
    user_ids: &[u32],
    leading_skip: usize,
    budget: usize,
    mut decode: F,
) -> Result<Selection, &'static str>
where
    F: FnMut(&[u32]) -> Vec<u8>,
{
    if budget == 0 || budget > MAX_PREFIX_TOKENS {
        return Err("prefix token budget must be in 1..=256");
    }
    let started = Instant::now();
    let count = user_ids.len().min(budget);
    let decoded = decode(&user_ids[..count]);
    if leading_skip > decoded.len() {
        return Err("template boundary exceeds decoded prefix");
    }
    let prefix = decoded[leading_skip..].to_vec();
    let mut inspected = 0;
    let mut decision = Decision {
        kind: Kind::Unknown,
        k: 3,
    };
    if prefix.len() <= MAX_PREFIX_BYTES {
        let complete_utf8 = match std::str::from_utf8(&prefix) {
            Ok(text) => Some(text),
            Err(error) if error.error_len().is_none() => {
                std::str::from_utf8(&prefix[..error.valid_up_to()]).ok()
            }
            Err(_) => None,
        };
        if let Some(mut text) = complete_utf8 {
            // Do not peek at another token to finish a possibly partial word.
            // Treat an exactly exhausted budget identically with or without a
            // longer unseen tail.
            if count == budget && text.as_bytes().last().is_some_and(u8::is_ascii_alphabetic) {
                text = text
                    .rfind(char::is_whitespace)
                    .map_or("", |index| &text[..index]);
            }
            inspected = text.len();
            decision = select_depth(text, PROFILE, 4);
        }
    }
    Ok(Selection {
        decision,
        token_count: count,
        decoded_bytes: prefix.len(),
        inspected_bytes: inspected,
        elapsed_ns: started.elapsed().as_nanos() as u64,
        prefix,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(ids: &[u32]) -> Vec<u8> {
        ids.iter().map(|&id| id as u8).collect()
    }

    fn ids(text: &str) -> Vec<u32> {
        text.bytes().map(u32::from).collect()
    }

    #[test]
    fn decoder_receives_only_the_first_budgeted_tokens_once() {
        let input: Vec<_> = (0..16_384).collect();
        let mut calls = 0;
        let selection = select(&input, 0, 64, |prefix| {
            calls += 1;
            assert_eq!(prefix, &input[..64]);
            b"Explain this design in prose.".to_vec()
        })
        .unwrap();
        assert_eq!(calls, 1);
        assert_eq!(selection.token_count, 64);
        assert_eq!(selection.decision.k, 2);
    }

    #[test]
    fn unseen_tail_and_total_length_do_not_change_the_decision() {
        let prefix = "Implement a Python function.                                ";
        let first = ids(&format!("{prefix} Explain it instead."));
        let second = ids(&format!(
            "{prefix}{}",
            " Ignore the earlier request.".repeat(1000)
        ));
        let budget = prefix.len();
        let a = select(&first, 0, budget, bytes).unwrap();
        let b = select(&second, 0, budget, bytes).unwrap();
        assert_eq!(a.prefix, b.prefix);
        assert_eq!(a.decision, b.decision);
        assert_eq!(a.token_count, budget);
    }

    #[test]
    fn missing_task_in_prefix_keeps_fixed_three() {
        let prefix = "<reference>\nBackground facts about a queue and its records.\n";
        let input = ids(&format!(
            "{prefix}</reference>\nImplement a Python function."
        ));
        let selected = select(&input, 0, prefix.len(), bytes).unwrap();
        assert_eq!(selected.decision.k, 3);
    }

    #[test]
    fn template_overlap_is_removed_without_reading_another_token() {
        let selected = select(&[1, 2], 1, 64, |_| b"\nExplain the design.".to_vec()).unwrap();
        assert_eq!(selected.prefix, b"Explain the design.");
        assert_eq!(selected.decision.k, 2);
    }

    #[test]
    fn partial_utf8_and_exhausted_words_never_extend_the_budget() {
        let selected = select(&[1], 0, 64, |_| b"Explain the design. \xf0".to_vec()).unwrap();
        assert_eq!(selected.decision.k, 2);
        let selected = select(&ids("Write code"), 0, 10, bytes).unwrap();
        assert_eq!(selected.inspected_bytes, 5);
    }

    #[test]
    fn invalid_or_oversized_prefix_falls_back_and_invalid_budget_fails() {
        assert_eq!(select(&[1], 0, 64, |_| vec![0xff]).unwrap().decision.k, 3);
        assert_eq!(
            select(&[1], 0, 64, |_| vec![b'a'; MAX_PREFIX_BYTES + 1])
                .unwrap()
                .decision
                .k,
            3
        );
        assert!(select(&[1], 0, 0, bytes).is_err());
        assert!(select(&[1], 0, MAX_PREFIX_TOKENS + 1, bytes).is_err());
    }
}
