//! memra-tokenizer — host-only GPT-2/BPE tokenizer (encode + decode + chat template).
//!
//! Algorithm TAKEn ~1:1 from llama.cpp's GPT-2 BPE path (`src/llama-vocab.cpp`,
//! `src/unicode.cpp`), Rust glue hand-rolled. Built from the model's own GGUF
//! tokenizer metadata (`tokenizer.ggml.*`) so it is integer-exact for that model.
//!
//! Scope: the `gpt2` vocab model with the `qwen35`/`qwen2`/`deepseek-v3`/`glm4` pre-tokenizers,
//! plus the `gemma4` SPM-style path — see `SUPPORTED_PRETOKENIZERS`. A model declaring anything else
//! is REFUSED at load (`UnknownPretokenizer`), because an unported pre-tokenizer produces
//! fluent output with wrong token ids and nothing downstream can see it.

pub mod chat;
pub mod json;
mod unicode;
mod unicode_data;

pub use chat::apply_chat_template_str;

use memra_gguf::{GgufFile, MetaValue};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// ggml token_type values (llama.cpp `LLAMA_TOKEN_TYPE_*`).
const TT_UNKNOWN: i64 = 2;
const TT_CONTROL: i64 = 3;
const TT_USER_DEFINED: i64 = 4;
const TT_BYTE: i64 = 6;
/// Dense token tables are indexed by token id; cap sparse metadata before it can
/// request a multi-gigabyte `Vec` for one distant id.
const MAX_TOKENIZER_ID: u32 = 1_000_000;
const MAX_TOKENIZER_SPARSE_FACTOR: usize = 16;
const MAX_TOKENIZER_SPARSE_SLACK: usize = 4096;
const QWEN35_PRETOKENIZE_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
/// llama.cpp `LLAMA_VOCAB_PRE_TYPE_QWEN2`. Differs from qwen35 in exactly two places —
/// `\p{L}+` vs `[\p{L}\p{M}]+` and `[^\s\p{L}\p{N}]+` vs `[^\s\p{L}\p{M}\p{N}]+` — both of
/// which the qwen35 state machine covers (see the `"qwen2"` arm in `PreSplit::resolve`).
const QWEN2_PRETOKENIZE_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
/// The three `Split` steps of the `deepseek-v3` (DEEPSEEK3_LLM) pre-tokenizer Sequence, in the
/// order HF serializes them: `\p{N}{1,3}` digit grouping, an isolated CJK/kana pass, then the
/// six-alternative pattern. `unicode::split_deepseek_v3` is a pass-for-pass port of exactly
/// these three. Read off the Hy3 and Step-3.7-Flash checkpoints' own `tokenizer.json`.
const DEEPSEEK_V3_SPLIT_REGEXES: [&str; 3] = [
    r"\p{N}{1,3}",
    "[\u{4e00}-\u{9fa5}\u{3040}-\u{309f}\u{30a0}-\u{30ff}]+",
    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
];

/// llama.cpp `LLAMA_VOCAB_PRE_TYPE_CHATGLM4` (`tokenizer.ggml.pre` = `glm4`), read off
/// zai-org/GLM-5.3-Flash @ 04c4e9e9's own `tokenizer.json` (sha256 19e77364…, the sha banked
/// in that lane's `artifact.lock`). Differs from `QWEN2_PRETOKENIZE_REGEX` in EXACTLY ONE
/// ATOM — `\p{N}{1,3}` instead of `\p{N}`, so digit runs group up to three (the cl100k /
/// LLaMA-3 convention) instead of one token per digit — and is character-identical
/// everywhere else. It still needs its OWN state machine (`unicode::split_glm4`) rather than
/// a parameterized `split_qwen35`: `split_qwen35` implements qwen35's mark-folding classes
/// (`[\p{L}\p{M}]+`, `[^\s\p{L}\p{M}\p{N}]+`), which the literal classes here do not, and the
/// two disagree on any text carrying combining marks.
const GLM4_PRETOKENIZE_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Every `tokenizer.ggml.pre` id memra implements an EXACT split for. This is the allowlist a
/// load is checked against and the list quoted in the load error, so the two can never drift.
pub const SUPPORTED_PRETOKENIZERS: &[&str] = &["qwen35", "qwen2", "deepseek-v3", "gemma4", "glm4"];

/// Escape hatch for deliberate experimentation with a family whose pre-tokenizer is not ported
/// yet. Set to `1` to downgrade the hard load error to a loud per-load WARN.
pub const ALLOW_UNKNOWN_PRETOKENIZER_ENV: &str = "MEMRA_ALLOW_UNKNOWN_PRETOKENIZER";

fn allow_unknown_pretokenizer() -> bool {
    std::env::var(ALLOW_UNKNOWN_PRETOKENIZER_ENV).as_deref() == Ok("1")
}

/// Serializes every TEST that touches `MEMRA_ALLOW_UNKNOWN_PRETOKENIZER`. The variable is
/// PROCESS-wide and `cargo test` runs the suite in parallel threads of ONE process, so
/// `pretokenizer_tests::opt_out_env_gate`'s `set_var("1")` was visible to
/// `hf_tests::hf_dir_refuses_unidentifiable_pretokenizer` while it ran: that test calls
/// `from_hf_dir`, which reads this same variable through `allow_unknown_pretokenizer` above, so
/// it LOADED instead of refusing and the suite failed with the qwen35-fallback WARN in its
/// stdout. It reddened at whatever interleaving the scheduler happened to pick and passed every
/// time it was run alone, which is why it read as noise. `opt_out_env_gate`'s old SAFETY note
/// asserted "no other test reads this variable", which was simply untrue: the prose was the
/// entire justification and nothing checked it against an actual reader. It lives at the crate
/// root because the two racing tests are in two different `#[cfg(test)]` modules.
/// (Found by lane/real-system-fingerprint-20260901, whose `cargo test --workspace` it reddened.)
#[cfg(test)]
static PRETOKENIZER_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Take the env lock, tolerating poisoning: a panic inside one of these tests must not convert
/// the others into a second, misleading failure that hides the first.
#[cfg(test)]
fn pretokenizer_env_lock() -> std::sync::MutexGuard<'static, ()> {
    PRETOKENIZER_ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// A model declared a pre-tokenizer memra has no exact split for.
///
/// Before 2026-08-19 this was one `eprintln!` per process followed by a silent fall-through to
/// the qwen35 split: the model loaded, generated fluent text, and every token id was wrong —
/// the same fluent-and-invisible class as the GGUF chat-template mint trap. Wrong ids poison
/// goldens, parity fixtures, acceptance counts and every quality number downstream, and nothing
/// in the stack can detect it after the fact. So it is a hard load error now.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownPretokenizer {
    /// The value that was rejected (`tokenizer.ggml.pre`, or `default` when an HF checkpoint's
    /// pre-tokenizer regexes matched no known family).
    pub pre: String,
    /// True when the vocab model is SPM-style (`tokenizer.ggml.model == "gemma4"`); a `pre`/model
    /// disagreement is itself the fault, so it is worth naming.
    pub spm_style: bool,
}

impl std::fmt::Display for UnknownPretokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "unsupported tokenizer.ggml.pre '{}' (vocab model is {}) — memra has no exact \
             pre-tokenizer split for it and token ids would NOT be exact. Supported: {}. \
             Set {}=1 to load anyway for deliberate experimentation (token ids will be wrong).",
            self.pre,
            if self.spm_style { "SPM/gemma4" } else { "gpt2" },
            SUPPORTED_PRETOKENIZERS.join(", "),
            ALLOW_UNKNOWN_PRETOKENIZER_ENV,
        )
    }
}

impl std::error::Error for UnknownPretokenizer {}

/// The pre-tokenizer split a loaded `Tokenizer` runs. Constructed only through
/// `PreSplit::resolve`, so "we do not know how to split for this model" is not a state a live
/// tokenizer can be in unless the operator asked for it via the env opt-out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreSplit {
    /// `unicode::split_qwen35` — serves both `qwen35` and `qwen2`.
    Qwen35,
    /// `unicode::split_deepseek_v3` — DeepSeek-V3 and the Step-3.5/3.7-Flash family.
    DeepseekV3,
    /// `unicode::split_glm4` — the zai-org GLM-4.x / GLM-5.x line (llama.cpp
    /// `LLAMA_VOCAB_PRE_TYPE_CHATGLM4`). qwen2's pattern with `\p{N}{1,3}` digit grouping.
    Glm4,
    /// gemma4 SPM-style BPE: `bpe_tokenize` splits whole lines itself and the `pre` id is never
    /// consulted. Requires the `gemma4` vocab model, not just the `pre` string.
    Spm,
    /// `MEMRA_ALLOW_UNKNOWN_PRETOKENIZER=1` was set for an unrecognized `pre`. Runs the qwen35
    /// split; token ids are NOT exact and every downstream measurement is invalid.
    UnknownFallbackQwen35,
}

impl PreSplit {
    /// Resolve a `tokenizer.ggml.pre` id against the implemented splits. `spm_style` is
    /// `tokenizer.ggml.model == "gemma4"`.
    pub fn resolve(pre: &str, spm_style: bool) -> Result<Self, UnknownPretokenizer> {
        Self::resolve_with(pre, spm_style, allow_unknown_pretokenizer())
    }

    /// `resolve` with the env decision passed in, so tests exercise both branches without
    /// mutating process-global environment underneath the rest of the suite.
    fn resolve_with(
        pre: &str,
        spm_style: bool,
        allow_unknown: bool,
    ) -> Result<Self, UnknownPretokenizer> {
        // The pair is matched, not just the `pre` string: an SPM vocab with a gpt2 `pre` (or the
        // reverse) is a metadata disagreement, and picking either side of it silently is how a
        // wrong split gets chosen for a right-looking model.
        match (pre, spm_style) {
            ("qwen35" | "qwen2", false) => Ok(PreSplit::Qwen35),
            ("deepseek-v3", false) => Ok(PreSplit::DeepseekV3),
            ("glm4", false) => Ok(PreSplit::Glm4),
            ("gemma4", true) => Ok(PreSplit::Spm),
            _ => {
                let err = UnknownPretokenizer {
                    pre: pre.to_string(),
                    spm_style,
                };
                if allow_unknown {
                    // Deliberately NOT once-per-process: this prints on every load so it cannot
                    // scroll out of one boot log and be missed on the next.
                    eprintln!(
                        "memra-tokenizer: WARNING {ALLOW_UNKNOWN_PRETOKENIZER_ENV}=1 — loading \
                         with {err} FALLING BACK to the qwen35 split. Token ids are NOT exact: \
                         goldens, parity fixtures, acceptance counts and quality numbers taken \
                         on this model are all invalid."
                    );
                    Ok(PreSplit::UnknownFallbackQwen35)
                } else {
                    Err(err)
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokAttr {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Byte,
    Other,
}

impl TokAttr {
    fn from_toktype(t: i64) -> Self {
        match t {
            TT_UNKNOWN => TokAttr::Unknown,
            TT_CONTROL => TokAttr::Control,
            TT_USER_DEFINED => TokAttr::UserDefined,
            TT_BYTE => TokAttr::Byte,
            1 => TokAttr::Normal,
            _ => TokAttr::Other,
        }
    }
    /// Tokens that participate in `tokenizer_st_partition` (special-token splitting):
    /// CONTROL | USER_DEFINED | UNKNOWN.
    fn is_special(self) -> bool {
        matches!(
            self,
            TokAttr::Control | TokAttr::UserDefined | TokAttr::Unknown
        )
    }
}

pub struct Tokenizer {
    /// id -> raw vocab piece string (byte-encoded GPT-2 form, e.g. "Ġworld").
    id_to_token: Vec<String>,
    /// piece string -> id.
    token_to_id: HashMap<String, u32>,
    /// per-token attribute.
    attrs: Vec<TokAttr>,
    /// (left, right) merge pair -> rank (lower = higher priority).
    bpe_ranks: HashMap<(String, String), i32>,
    /// special-token ids, sorted by descending piece length (llama's cache order).
    special_tokens: Vec<u32>,
    eos_id: u32,
    /// EVERY end-of-generation id the checkpoint declares, `eos_id` first.
    ///
    /// `eos_id` alone is a lie on any vendor that ships an ARRAY `eos_token_id`: GLM-5.3-Flash
    /// declares `[154820 <|endoftext|>, 154827 <|user|>, 154829 <|observation|>]` in
    /// `generation_config.json` (its turn boundaries — the assistant reaching either of the
    /// last two has left its turn), and the serve path stopped on the first id only, so a
    /// finished answer ran on into a hallucinated multi-turn transcript and reported
    /// `finish_reason: "length"` (research/glm53-flash-bringup-20260827, 2026-08-27).
    /// GGUF keeps exactly one id here (its metadata declares one), so no GGUF family moves.
    eos_ids: Vec<u32>,
    bos_id: Option<u32>,
    add_bos: bool,
    pre: String,
    /// The split `pre` resolved to at load. Kept alongside the raw `pre` string so the encode
    /// path never re-interprets metadata and has no "unknown" arm to fall through.
    split: PreSplit,
    chat_template: Option<String>,
    /// SPM-style BPE (gemma4): \u2581 whitespace escaping, raw-UTF-8 merges, <0xXX> byte fallback.
    spm_style: bool,
    /// deepseek-v4 encoding revision, detected from the checkpoint's config.json dspark_*
    /// key census at `from_hf_dir` (see `chat::Dsv4Encoding` \u2014 the effort ladder differs
    /// between the preview and 0731 checkpoints while every tokenizer/template byte is
    /// identical). None = unknown (no config.json next to the tokenizer, or a GGUF \u2014 no
    /// dsv4 GGUF lineage exists yet and no metadata key is defined for it); rendering then
    /// refuses dsv4 effort levels whose bytes differ across revisions instead of guessing.
    dsv4_encoding: Option<chat::Dsv4Encoding>,
}

/// A bigram in the BPE work queue. Ordering matches llama.cpp's comparator:
/// the priority_queue pops the *smallest* (rank, left) under the std comparator
/// `l.rank > r.rank || (l.rank == r.rank && l.left > r.left)`. We implement `Ord`
/// so a max-heap pops that same element (min rank, then min left).
#[derive(Clone, Eq, PartialEq)]
struct Bigram {
    left: i32,
    right: i32,
    rank: i32,
    text: String,
}

impl Ord for Bigram {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap; we want the element with the lowest rank
        // (ties: lowest left index) to be "greatest" so it pops first.
        match other.rank.cmp(&self.rank) {
            Ordering::Equal => other.left.cmp(&self.left),
            o => o,
        }
    }
}
impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A symbol (one or more codepoints) in the BPE chain. Mirrors `llm_symbol`.
struct Symbol {
    text: String,
    prev: i32,
    next: i32,
    n: usize, // codepoint count (0 == merged away)
}

impl Tokenizer {
    /// Build a tokenizer from a model's GGUF tokenizer metadata.
    pub fn from_gguf(g: &GgufFile) -> Result<Self, String> {
        let model = g
            .metadata
            .get("tokenizer.ggml.model")
            .and_then(|v| v.as_str())
            .ok_or("missing tokenizer.ggml.model")?;
        if model != "gpt2" && model != "gemma4" {
            return Err(format!(
                "unsupported tokenizer model '{model}' (only gpt2/gemma4)"
            ));
        }
        // gemma4 = SPM-style BPE (llama-vocab.cpp): spaces escaped to \u2581 by the normalizer,
        // merges over raw UTF-8 (NO gpt2 byte-encoding), whole-line pre-split, <0xXX> byte
        // fallback tokens, add_bos force-true (PR #21500 workaround).
        let spm_style = model == "gemma4";
        let pre = g
            .metadata
            .get("tokenizer.ggml.pre")
            .and_then(|v| v.as_str())
            .unwrap_or(if spm_style { "gemma4" } else { "default" })
            .to_string();
        // Resolve BEFORE any of the (expensive) vocab/merge work: an unsupported pre-tokenizer
        // is a load refusal, not a warning, so there is no reason to build the tables first.
        let split = PreSplit::resolve(&pre, spm_style).map_err(|e| e.to_string())?;

        // tokens[]
        let tokens = match g.metadata.get("tokenizer.ggml.tokens") {
            Some(MetaValue::Array(a)) => a,
            _ => return Err("missing tokenizer.ggml.tokens array".into()),
        };
        let n = tokens.len();
        if n > MAX_TOKENIZER_ID as usize + 1 {
            return Err(format!(
                "tokenizer.ggml.tokens has {n} entries; maximum supported vocabulary is {}",
                MAX_TOKENIZER_ID + 1
            ));
        }
        let mut id_to_token = Vec::with_capacity(n);
        let mut token_to_id = HashMap::with_capacity(n);
        for (i, t) in tokens.iter().enumerate() {
            let s = t.as_str().ok_or("non-string in tokens[]")?.to_string();
            // first-id-wins on duplicates (llama keeps the map's first insert)
            token_to_id.entry(s.clone()).or_insert(i as u32);
            id_to_token.push(s);
        }

        // token_type[] -> attrs
        let mut attrs = vec![TokAttr::Normal; n];
        if let Some(MetaValue::Array(a)) = g.metadata.get("tokenizer.ggml.token_type") {
            for (i, v) in a.iter().enumerate().take(n) {
                if let Some(t) = v.as_u64() {
                    attrs[i] = TokAttr::from_toktype(t as i64);
                } else if let MetaValue::I32(t) = v {
                    attrs[i] = TokAttr::from_toktype(*t as i64);
                }
            }
        }

        // merges[] -> ranks. Each entry is "first second" (split on first space at idx>=1).
        let mut bpe_ranks = HashMap::new();
        if let Some(MetaValue::Array(a)) = g.metadata.get("tokenizer.ggml.merges") {
            for (i, v) in a.iter().enumerate() {
                let word = v.as_str().ok_or("non-string in merges[]")?;
                // llama: pos = word.find(' ', 1) — a *byte* search starting at byte 1.
                // (The space separating the two pieces is always single-byte ASCII; the
                // pieces themselves may contain multibyte chars like 'Ġ', so we search bytes.)
                let bytes = word.as_bytes();
                if let Some(pos) = bytes.iter().skip(1).position(|&b| b == b' ').map(|p| p + 1) {
                    let first = word[..pos].to_string();
                    let second = word[pos + 1..].to_string();
                    bpe_ranks.insert((first, second), i as i32);
                }
            }
        } else {
            return Err("missing tokenizer.ggml.merges array".into());
        }

        // special-token cache: CONTROL|USER_DEFINED|UNKNOWN, sorted by descending text length.
        let mut special_tokens: Vec<u32> = (0..n as u32)
            .filter(|&id| attrs[id as usize].is_special())
            .collect();
        special_tokens.sort_by(|&a, &b| {
            id_to_token[b as usize]
                .len()
                .cmp(&id_to_token[a as usize].len())
        });

        let eos_id = g
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .ok_or("missing tokenizer.ggml.eos_token_id")?;
        let bos_id = g
            .metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);
        let add_bos = g
            .metadata
            .get("tokenizer.ggml.add_bos_token")
            .and_then(|v| match v {
                MetaValue::Bool(b) => Some(*b),
                _ => v.as_u64().map(|x| x != 0),
            })
            .unwrap_or(false);
        let add_bos = add_bos || spm_style;

        let chat_template = g
            .metadata
            .get("tokenizer.chat_template")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        Ok(Tokenizer {
            id_to_token,
            token_to_id,
            attrs,
            bpe_ranks,
            special_tokens,
            eos_id,
            // GGUF declares ONE eos id (`tokenizer.ggml.eos_token_id`); there is no array
            // form to honour, so every GGUF family's stop set is byte-identical to before.
            eos_ids: vec![eos_id],
            bos_id,
            add_bos,
            pre,
            split,
            chat_template,
            spm_style,
            // No dsv4 GGUF lineage exists (dsv4 serves from safetensors dirs); when a mint
            // lane defines one, it must carry the encoding revision in GGUF metadata —
            // unknown here means dsv4 "high"/"max" renders refuse rather than guess.
            dsv4_encoding: None,
        })
    }

    /// Build a tokenizer from an HF fast-tokenizer checkpoint directory
    /// (`tokenizer.json` + optional `tokenizer_config.json` / `generation_config.json` /
    /// `chat_template.jinja`). Only byte-level BPE (the gpt2 class — MiniMax-M3, Qwen,
    /// Llama-3 style) is supported: `model.type == "BPE"` with a ByteLevel pre-tokenizer.
    ///
    /// Mapping to the GGUF-built struct:
    ///   - model.vocab (token -> id map)             -> id_to_token / token_to_id
    ///   - model.merges ("a b" strings OR [a,b] pairs; both HF serializations) -> bpe_ranks
    ///   - added_tokens special=true -> Control class (split before BPE + hidden on decode);
    ///     non-special added tokens stay Normal.
    ///   - eos/bos: tokenizer_config eos_token/bos_token (string or {content} object),
    ///     generation_config eos_token_id (int or array) as the eos fallback.
    ///   - add_bos: tokenizer_config add_bos_token (default false).
    ///   - chat template: tokenizer_config chat_template, else chat_template.jinja.
    ///   - pre-tokenizer: `tokenizer_config.pretokenize_regex`, else the `Split` step regexes of
    ///     `tokenizer.json`'s own `pre_tokenizer`, matched byte-exactly against the qwen35 /
    ///     qwen2 / deepseek-v3 constants. No match -> a hard error naming both observations.
    pub fn from_hf_dir(dir: &std::path::Path) -> Result<Self, String> {
        let tj_path = dir.join("tokenizer.json");
        let text = std::fs::read_to_string(&tj_path)
            .map_err(|e| format!("read {}: {e}", tj_path.display()))?;
        let tj = json::parse(&text).map_err(|e| format!("{}: {e}", tj_path.display()))?;

        let model = tj.get("model").ok_or("tokenizer.json: missing model")?;
        if let Some(t) = model.get("type").and_then(|v| v.as_str())
            && t != "BPE"
        {
            return Err(format!(
                "unsupported tokenizer.json model type '{t}' (only BPE)"
            ));
        }
        // byte-level check: pre_tokenizer.type == ByteLevel (possibly inside a Sequence).
        let pre_tok = tj
            .get("pre_tokenizer")
            .ok_or("tokenizer.json: missing pre_tokenizer")?;
        if !pre_tokenizer_is_byte_level(pre_tok) {
            return Err(
                "tokenizer.json: pre_tokenizer is not ByteLevel — only byte-level \
                        BPE is supported"
                    .into(),
            );
        }

        // ---- vocab (token -> id). ids may exceed the map len (added_tokens append). ----
        let vocab = model
            .get("vocab")
            .and_then(|v| v.as_obj())
            .ok_or("tokenizer.json: missing model.vocab")?;
        let empty: Vec<json::Value> = Vec::new();
        let added = tj
            .get("added_tokens")
            .and_then(|v| v.as_arr())
            .unwrap_or(&empty);
        let mut max_id = 0u32;
        let mut used_ids = HashSet::new();
        for v in vocab.values() {
            let id = v
                .as_u64()
                .ok_or("tokenizer.json: non-integer id in model.vocab")?;
            let id = u32::try_from(id).map_err(|_| "tokenizer.json: vocabulary id exceeds u32")?;
            if id > MAX_TOKENIZER_ID {
                return Err(format!(
                    "tokenizer.json: vocabulary id {id} exceeds maximum {MAX_TOKENIZER_ID}"
                ));
            }
            max_id = max_id.max(id);
            used_ids.insert(id);
        }
        for a in added {
            let id = a
                .get("id")
                .and_then(|v| v.as_u64())
                .ok_or("tokenizer.json: added_tokens entry missing id")?;
            let id = u32::try_from(id).map_err(|_| "tokenizer.json: added token id exceeds u32")?;
            if id > MAX_TOKENIZER_ID {
                return Err(format!(
                    "tokenizer.json: added token id {id} exceeds maximum {MAX_TOKENIZER_ID}"
                ));
            }
            max_id = max_id.max(id);
            used_ids.insert(id);
        }
        let n = (max_id as usize)
            .checked_add(1)
            .ok_or("tokenizer.json: vocabulary size overflow")?;
        let entry_count = used_ids.len();
        let max_dense_len = entry_count
            .saturating_mul(MAX_TOKENIZER_SPARSE_FACTOR)
            .saturating_add(MAX_TOKENIZER_SPARSE_SLACK);
        if n > max_dense_len {
            return Err(format!(
                "tokenizer.json: vocabulary ids are too sparse (dense length {n}, {} entries; maximum {max_dense_len})",
                entry_count
            ));
        }
        let mut id_to_token = vec![String::new(); n];
        let mut token_to_id: HashMap<String, u32> = HashMap::with_capacity(n);
        let mut attrs = vec![TokAttr::Normal; n];
        for (tok, v) in vocab {
            let id = u32::try_from(
                v.as_u64()
                    .ok_or("tokenizer.json: non-integer id in model.vocab")?,
            )
            .map_err(|_| "tokenizer.json: vocabulary id exceeds u32")?;
            id_to_token[id as usize] = tok.clone();
            token_to_id.entry(tok.clone()).or_insert(id);
        }
        // added_tokens: register content + special flag. special=true -> Control (the class
        // that is split out before BPE and hidden by decode_special(.., false)).
        for a in added {
            let id = u32::try_from(
                a.get("id")
                    .and_then(|v| v.as_u64())
                    .ok_or("tokenizer.json: added_tokens entry missing id")?,
            )
            .map_err(|_| "tokenizer.json: added token id exceeds u32")?;
            let content = a
                .get("content")
                .and_then(|v| v.as_str())
                .ok_or("tokenizer.json: added_tokens entry missing content")?;
            if id_to_token[id as usize].is_empty() {
                id_to_token[id as usize] = content.to_string();
            }
            token_to_id.entry(content.to_string()).or_insert(id);
            if a.get("special").and_then(|v| v.as_bool()).unwrap_or(false) {
                attrs[id as usize] = TokAttr::Control;
            } else {
                // HF's AddedVocabulary matches EVERY added token whole (special or not) before
                // the BPE model runs; `special` only controls skip_special_tokens on decode.
                // UserDefined = split whole before BPE but NOT hidden on decode — exactly the
                // HF non-special class (Hy3's `<think:opensource>`/`<｜reasoning_mode…｜>` chat
                // tokens are special=false and MUST encode as single ids, 2026-07-09).
                attrs[id as usize] = TokAttr::UserDefined;
            }
        }

        // ---- merges: array of "a b" strings OR [a, b] pairs (HF emits both). ----
        let merges = model
            .get("merges")
            .and_then(|v| v.as_arr())
            .ok_or("tokenizer.json: missing model.merges")?;
        let mut bpe_ranks = HashMap::with_capacity(merges.len());
        for (i, m) in merges.iter().enumerate() {
            let (first, second) = match m {
                json::Value::Str(s) => {
                    // byte search for the separating space from byte 1 (same as the GGUF
                    // path: pieces may contain multibyte chars like 'Ġ', the space is ASCII).
                    let bytes = s.as_bytes();
                    let pos = bytes
                        .iter()
                        .skip(1)
                        .position(|&b| b == b' ')
                        .map(|p| p + 1)
                        .ok_or_else(|| format!("tokenizer.json: merges[{i}] has no space"))?;
                    (s[..pos].to_string(), s[pos + 1..].to_string())
                }
                json::Value::Arr(a) if a.len() == 2 => {
                    let f = a[0]
                        .as_str()
                        .ok_or_else(|| format!("tokenizer.json: merges[{i}] non-string pair"))?;
                    let s2 = a[1]
                        .as_str()
                        .ok_or_else(|| format!("tokenizer.json: merges[{i}] non-string pair"))?;
                    (f.to_string(), s2.to_string())
                }
                _ => {
                    return Err(format!(
                        "tokenizer.json: merges[{i}] is neither \"a b\" string nor [a, b] pair"
                    ));
                }
            };
            bpe_ranks.insert((first, second), i as i32);
        }

        // special-token cache: same construction as from_gguf.
        let mut special_tokens: Vec<u32> = (0..n as u32)
            .filter(|&id| attrs[id as usize].is_special())
            .collect();
        special_tokens.sort_by(|&a, &b| {
            id_to_token[b as usize]
                .len()
                .cmp(&id_to_token[a as usize].len())
        });

        // ---- sidecars: tokenizer_config.json + generation_config.json ----
        let tc = std::fs::read_to_string(dir.join("tokenizer_config.json"))
            .ok()
            .and_then(|t| json::parse(&t).ok());
        let gc = std::fs::read_to_string(dir.join("generation_config.json"))
            .ok()
            .and_then(|t| json::parse(&t).ok());

        // eos_token/bos_token: plain string OR {"content": "..."} AddedToken object.
        let tok_content = |v: &json::Value| -> Option<String> {
            v.as_str().map(|s| s.to_string()).or_else(|| {
                v.get("content")
                    .and_then(|c| c.as_str())
                    .map(|s| s.to_string())
            })
        };
        let eos_from_cfg = tc
            .as_ref()
            .and_then(|c| c.get("eos_token"))
            .and_then(&tok_content)
            .and_then(|s| token_to_id.get(&s).copied());
        // generation_config eos_token_id: int or an ARRAY of ints, and the array is the
        // vendor's whole stop set — not a list whose first entry is the answer. Reading only
        // `a.first()` here is what let GLM-5.3-Flash run past `<|user|>` (lane
        // glm53-flash-bringup, 2026-08-27): ids 2 and 3 of a declared triple were dropped on
        // the floor. `eos_id` (the SCALAR, consumed by the boot log, embed pooling, the
        // grammar/tokrx bridges and every `eos_id()` caller) keeps its exact old selection —
        // tokenizer_config `eos_token` first, else the FIRST generation_config id — and the
        // rest of the declared ids ride `eos_ids` into `eog_ids()`.
        let gen_eos_ids: Vec<u32> = gc
            .as_ref()
            .and_then(|c| c.get("eos_token_id"))
            .map(|v| match v {
                json::Value::Num(_) => v.as_u64().map(|x| x as u32).into_iter().collect(),
                json::Value::Arr(a) => a
                    .iter()
                    .filter_map(|x| x.as_u64())
                    .map(|x| x as u32)
                    .collect(),
                _ => Vec::new(),
            })
            .unwrap_or_default();
        let eos_from_gen = gen_eos_ids.first().copied();
        let eos_id = eos_from_cfg.or(eos_from_gen).ok_or(
            "no eos token: need tokenizer_config.json eos_token or \
             generation_config.json eos_token_id",
        )?;
        // Declared order, eos_id first, no dedup survivors. An id past the vocab is kept as
        // declared rather than silently dropped: it is inert (nothing can sample an id the
        // logit row has no column for) and dropping it would hide a bad checkpoint.
        let mut eos_ids = vec![eos_id];
        for id in gen_eos_ids {
            if !eos_ids.contains(&id) {
                eos_ids.push(id);
            }
        }
        let bos_id = tc
            .as_ref()
            .and_then(|c| c.get("bos_token"))
            .and_then(&tok_content)
            .and_then(|s| token_to_id.get(&s).copied());
        let add_bos = tc
            .as_ref()
            .and_then(|c| c.get("add_bos_token"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // chat template: tokenizer_config chat_template string, else chat_template.jinja file.
        let chat_template = tc
            .as_ref()
            .and_then(|c| c.get("chat_template"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| std::fs::read_to_string(dir.join("chat_template.jinja")).ok());
        // Pre-tokenizer identification, in order of authority:
        //   1. `tokenizer_config.json`'s `pretokenize_regex` (Qwen ships it explicitly), and
        //   2. the `Split` step regexes of `tokenizer.json`'s own `pre_tokenizer`.
        // (2) was missing until 2026-08-19, so ONLY Qwen checkpoints could ever be identified
        // here and everything else fell to `default` -> the silent qwen35 fallback. The Hy3
        // checkpoints were being mis-tokenized that way while `split_deepseek_v3` — the exact
        // splitter their own tokenizer.json asks for — already shipped in this crate.
        let cfg_regex = tc
            .as_ref()
            .and_then(|c| c.get("pretokenize_regex"))
            .and_then(|v| v.as_str());
        let mut tj_regexes: Vec<String> = Vec::new();
        collect_split_regexes(pre_tok, &mut tj_regexes);
        let pre = cfg_regex
            .and_then(|r| pre_from_split_regexes(std::slice::from_ref(&r.to_string())))
            .or_else(|| pre_from_split_regexes(&tj_regexes))
            .unwrap_or("default");
        let split = PreSplit::resolve(pre, false).map_err(|e| {
            if pre == "default" {
                format!(
                    "{e}\n  (HF checkpoint {}: tokenizer_config.json pretokenize_regex = {:?}, \
                     tokenizer.json pre_tokenizer Split regexes = {:?} — neither matched a known \
                     family)",
                    dir.display(),
                    cfg_regex,
                    tj_regexes,
                )
            } else {
                e.to_string()
            }
        })?;

        // deepseek-v4 encoding revision from the checkpoint's own config.json (dspark_* key
        // census — the only artifact-level marker; tokenizer/template files are byte-identical
        // across the preview and 0731 checkpoints). Missing/unparseable config.json (e.g. a
        // tokenizer-only ref dir) = unknown; a PARTIAL dspark key set is a corrupt config and
        // refuses the load rather than guessing an effort ladder.
        let dsv4_encoding = dsv4_encoding_from_config(dir)?;

        Ok(Tokenizer {
            id_to_token,
            token_to_id,
            attrs,
            bpe_ranks,
            special_tokens,
            eos_id,
            eos_ids,
            bos_id,
            add_bos,
            pre: pre.to_string(),
            split,
            chat_template,
            spm_style: false,
            dsv4_encoding,
        })
    }

    pub fn eos_id(&self) -> u32 {
        self.eos_id
    }
    /// Every eos id the checkpoint declares, `eos_id` first. One entry for GGUF and for any
    /// checkpoint whose `generation_config.json` names a single id; the whole declared array
    /// otherwise. Prefer `eog_ids()` for a stop set — it adds the name-keyed backstop.
    pub fn eos_ids(&self) -> &[u32] {
        &self.eos_ids
    }
    /// Exact-piece id lookup (vision special tokens etc.). None = not in the vocab.
    pub fn id_of(&self, piece: &str) -> Option<u32> {
        self.token_to_id.get(piece).copied()
    }
    /// Every end-of-generation id the CHECKPOINT declares (`eos_ids`), plus the common
    /// turn-end control tokens present in the vocab (llama's special_eog set — <|im_end|>
    /// chatml, <turn|>/<end_of_turn> gemma) as a name-keyed backstop for checkpoints whose
    /// metadata names only one.
    ///
    /// This is the serve path's stop set (`worker::run` unions it into `GenParams::eos`).
    pub fn eog_ids(&self) -> Vec<u32> {
        let mut ids = self.eos_ids.clone();
        for t in ["<|im_end|>", "<turn|>", "<end_of_turn>"] {
            if let Some(&id) = self.token_to_id.get(t)
                && !ids.contains(&id)
            {
                ids.push(id);
            }
        }
        ids
    }
    pub fn bos_id(&self) -> Option<u32> {
        self.bos_id
    }
    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }
    pub fn pre(&self) -> &str {
        &self.pre
    }
    /// The split `pre` resolved to. `UnknownFallbackQwen35` means the env opt-out is engaged and
    /// this tokenizer's ids are NOT exact — a serve gate can refuse on it.
    pub fn split(&self) -> PreSplit {
        self.split
    }
    pub fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }
    /// deepseek-v4 encoding revision detected at load (config.json dspark_* census);
    /// None = unknown. Meaningful only for dsv4-template artifacts.
    pub fn dsv4_encoding(&self) -> Option<chat::Dsv4Encoding> {
        self.dsv4_encoding
    }

    #[inline]
    fn text_to_token(&self, s: &str) -> Option<u32> {
        self.token_to_id.get(s).copied()
    }

    fn find_bpe_rank(&self, left: &str, right: &str) -> i32 {
        self.bpe_ranks
            .get(&(left.to_string(), right.to_string()))
            .copied()
            .unwrap_or(-1)
    }

    /// Encode text -> token ids.
    ///
    /// `add_special` controls whether a BOS is prepended when the model asks for it.
    /// `parse_special` (always true here) splits control/user-defined/unknown tokens
    /// (e.g. `<|im_start|>`) out before BPE — matching llama's default tokenize().
    pub fn encode(&self, text: &str, add_special: bool) -> Vec<u32> {
        self.encode_special(text, add_special, true)
    }

    pub fn encode_special(&self, text: &str, add_special: bool, parse_special: bool) -> Vec<u32> {
        let mut output: Vec<u32> = Vec::new();
        if add_special
            && self.add_bos
            && let Some(b) = self.bos_id
        {
            output.push(b);
        }
        if text.is_empty() {
            return output;
        }

        // fragment buffer: alternate raw-text spans and resolved special-token ids.
        for frag in self.st_partition(text, parse_special) {
            match frag {
                Fragment::Token(id) => output.push(id),
                Fragment::Text(span) => self.bpe_tokenize(&span, &mut output),
            }
        }
        output
    }

    /// `tokenizer_st_partition` — split out special tokens (longest first) before BPE.
    fn st_partition(&self, text: &str, parse_special: bool) -> Vec<Fragment> {
        let mut frags = vec![Fragment::Text(text.to_string())];
        for &sid in &self.special_tokens {
            let attr = self.attrs[sid as usize];
            // when parse_special is false, skip CONTROL/UNKNOWN (user-defined still split).
            if !parse_special && matches!(attr, TokAttr::Control | TokAttr::Unknown) {
                continue;
            }
            let needle = &self.id_to_token[sid as usize];
            if needle.is_empty() {
                continue;
            }
            let mut next: Vec<Fragment> = Vec::with_capacity(frags.len());
            for f in frags.drain(..) {
                match f {
                    Fragment::Token(id) => next.push(Fragment::Token(id)),
                    Fragment::Text(s) => {
                        let mut rest: &str = &s;
                        let mut acc = String::new();
                        while let Some(m) = rest.find(needle.as_str()) {
                            acc.push_str(&rest[..m]);
                            if !acc.is_empty() {
                                next.push(Fragment::Text(std::mem::take(&mut acc)));
                            }
                            next.push(Fragment::Token(sid));
                            rest = &rest[m + needle.len()..];
                        }
                        acc.push_str(rest);
                        if !acc.is_empty() {
                            next.push(Fragment::Text(acc));
                        }
                    }
                }
            }
            frags = next;
        }
        frags
    }

    /// Core BPE over one raw-text fragment (`llm_tokenizer_bpe_session::tokenize`).
    fn bpe_tokenize(&self, text: &str, output: &mut Vec<u32>) {
        if self.spm_style {
            // gemma4 (llama PRE_TYPE_GEMMA4): escape spaces to \u2581 on the raw fragment,
            // split whole lines ([^\n]+|[\n]+), run BPE on raw UTF-8 chars.
            let escaped: String = text
                .chars()
                .map(|c| if c == ' ' { '\u{2581}' } else { c })
                .collect();
            let mut words: Vec<String> = Vec::new();
            let mut cur = String::new();
            let mut cur_nl: Option<bool> = None;
            for c in escaped.chars() {
                let nl = c == '\n';
                if cur_nl != Some(nl) && !cur.is_empty() {
                    words.push(std::mem::take(&mut cur));
                }
                cur_nl = Some(nl);
                cur.push(c);
            }
            if !cur.is_empty() {
                words.push(cur);
            }
            for word in &words {
                // newline-run fix (llama PR #21343): whole-word vocab hit short-circuits BPE.
                if word.chars().all(|c| c == '\n')
                    && let Some(tok) = self.text_to_token(word)
                {
                    output.push(tok);
                    continue;
                }
                self.bpe_merge_word(word, output);
            }
            return;
        }
        // 1) pre-tokenizer split, then 2) GPT-2 byte-encode each word.
        //
        // Exhaustive on `PreSplit` and has NO fall-through arm: the "we do not know how to split
        // this" case was resolved (and refused) at load, so it cannot arrive here. Adding a
        // `PreSplit` variant must fail to compile until this match handles it.
        let words: Vec<String> = match self.split {
            // qwen35 also serves qwen2: llama.cpp's qwen2 regex differs from qwen35's only in
            // [\p{L}\p{M}]+ vs \p{L}+, which the qwen35 state machine covers.
            PreSplit::Qwen35 => unicode::split_qwen35(text),
            // Step-3.5/3.7-Flash and the DeepSeek-V3 family
            // (llama.cpp LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM). Materially different from qwen2:
            // \p{N}{1,3} digit grouping, an isolated CJK/kana pass, and \p{P}/\p{S}-only runs.
            PreSplit::DeepseekV3 => unicode::split_deepseek_v3(text),
            // zai-org GLM-4.x / GLM-5.x (llama.cpp LLAMA_VOCAB_PRE_TYPE_CHATGLM4): qwen2's
            // pattern with `\p{N}{1,3}` digit grouping, and literal `\p{L}` letter runs that
            // do NOT fold combining marks the way the qwen35 machine does.
            PreSplit::Glm4 => unicode::split_glm4(text),
            // MEMRA_ALLOW_UNKNOWN_PRETOKENIZER=1 — the operator asked for wrong ids. The WARN
            // was printed at load; do not repeat it once per fragment.
            PreSplit::UnknownFallbackQwen35 => unicode::split_qwen35(text),
            // Unreachable: `spm_style` short-circuits above, and `PreSplit::Spm` is only
            // produced together with it.
            PreSplit::Spm => unreachable!("PreSplit::Spm implies spm_style, handled above"),
        };

        for word in &words {
            let word = unicode::byte_encode(word);
            self.bpe_merge_word(&word, output);
        }
    }

    /// BPE merge over one pre-split word (symbols = unicode chars), emitting token ids with
    /// byte fallback (gpt2 single-char byte tokens, or SPM <0xXX> tokens when spm_style).
    fn bpe_merge_word(&self, word: &str, output: &mut Vec<u32>) {
        {
            let word = word.to_string();

            // build the symbol chain, one symbol per unicode char initially.
            let chars: Vec<char> = word.chars().collect();
            let mut symbols: Vec<Symbol> = Vec::with_capacity(chars.len());
            for (i, &c) in chars.iter().enumerate() {
                symbols.push(Symbol {
                    text: c.to_string(),
                    prev: i as i32 - 1,
                    next: if i + 1 == chars.len() {
                        -1
                    } else {
                        i as i32 + 1
                    },
                    n: 1,
                });
            }

            // seed the work queue with adjacent bigrams.
            let mut queue: BinaryHeap<Bigram> = BinaryHeap::new();
            for i in 1..symbols.len() {
                self.add_bigram(&symbols, i as i32 - 1, i as i32, &mut queue);
            }

            // merge by rank.
            while let Some(bigram) = queue.pop() {
                let li = bigram.left as usize;
                let ri = bigram.right as usize;
                if symbols[li].n == 0 || symbols[ri].n == 0 {
                    continue;
                }
                let combined = format!("{}{}", symbols[li].text, symbols[ri].text);
                if combined != bigram.text {
                    continue; // outdated bigram
                }
                // merge right into left
                symbols[li].text = combined;
                symbols[li].n += symbols[ri].n;
                symbols[ri].n = 0;
                let r_next = symbols[ri].next;
                symbols[li].next = r_next;
                if r_next >= 0 {
                    symbols[r_next as usize].prev = bigram.left;
                }
                let l_prev = symbols[li].prev;
                let l_next = symbols[li].next;
                self.add_bigram(&symbols, l_prev, bigram.left, &mut queue);
                self.add_bigram(&symbols, bigram.left, l_next, &mut queue);
            }

            // emit final symbols in chain order, with byte-level fallback.
            for sym in &symbols {
                if sym.n == 0 {
                    continue;
                }
                match self.text_to_token(&sym.text) {
                    Some(tok) => output.push(tok),
                    None => {
                        // byte fallback: each *byte* of the piece must be its own token.
                        for b in sym.text.bytes() {
                            let bs = if self.spm_style {
                                format!("<0x{b:02X}>") // SPM-style byte tokens (gemma4)
                            } else {
                                (b as char).to_string()
                            };
                            if let Some(t) = self.text_to_token(&bs) {
                                output.push(t);
                            }
                        }
                    }
                }
            }
        }
    }

    fn add_bigram(
        &self,
        symbols: &[Symbol],
        left: i32,
        right: i32,
        queue: &mut BinaryHeap<Bigram>,
    ) {
        if left == -1 || right == -1 {
            return;
        }
        let lt = &symbols[left as usize].text;
        let rt = &symbols[right as usize].text;
        let rank = self.find_bpe_rank(lt, rt);
        if rank < 0 {
            return;
        }
        queue.push(Bigram {
            left,
            right,
            rank,
            text: format!("{lt}{rt}"),
        });
    }

    /// Decode token ids -> String. `special=false` drops control tokens (chat tags);
    /// `special=true` renders them as their literal text.
    pub fn decode(&self, ids: &[u32]) -> String {
        self.decode_special(ids, true)
    }

    /// True for Control/Unknown tokens — vocab entries that are protocol markers, not text.
    /// External vocab consumers (llguidance's toktrie, constrained decoding) must not let a
    /// grammar match these as literal bytes (a JSON string could otherwise smuggle
    /// `<|im_start|>`); they substitute a non-text marker form instead.
    pub fn token_is_control(&self, id: u32) -> bool {
        matches!(
            self.attrs.get(id as usize),
            Some(TokAttr::Control) | Some(TokAttr::Unknown)
        )
    }

    /// True iff `id` is in the special set `st_partition` splits out of raw text (control /
    /// user-defined / unknown) — i.e. an id a plain-text LITERAL can produce during encode.
    /// The vision special-id intake guard (memra-server) keys on this: an ordinary vocab
    /// entry that merely looks like a marker is honest text and must not be policed.
    pub fn token_is_special(&self, id: u32) -> bool {
        self.special_tokens.contains(&id)
    }

    pub fn decode_special(&self, ids: &[u32], special: bool) -> String {
        String::from_utf8_lossy(&self.decode_bytes_special(ids, special)).into_owned()
    }

    /// Decode token ids to their exact byte stream. Streaming callers must retain incomplete
    /// UTF-8 suffixes across token boundaries instead of replacing them prematurely.
    pub fn decode_bytes_special(&self, ids: &[u32], special: bool) -> Vec<u8> {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            let i = id as usize;
            if i >= self.id_to_token.len() {
                continue;
            }
            let attr = self.attrs[i];
            let piece = &self.id_to_token[i];
            match attr {
                TokAttr::Normal | TokAttr::Byte => {
                    if self.spm_style {
                        // gemma4: <0xXX> byte tokens -> raw byte; else unescape \u2581 -> space.
                        if (matches!(attr, TokAttr::Byte)
                            || (piece.len() == 6
                                && piece.starts_with("<0x")
                                && piece.ends_with('>')))
                            && let Ok(b) = u8::from_str_radix(&piece[3..5], 16)
                        {
                            bytes.push(b);
                            continue;
                        }
                        for c in piece.chars() {
                            if c == '\u{2581}' {
                                bytes.push(b' ');
                            } else {
                                let mut buf = [0u8; 4];
                                bytes.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                            }
                        }
                    } else {
                        // undo GPT-2 byte encoding: each char -> one raw byte.
                        self.piece_to_bytes(piece, &mut bytes);
                    }
                }
                TokAttr::UserDefined => {
                    // user-defined tokens are literal text (not byte-encoded).
                    bytes.extend_from_slice(piece.as_bytes());
                }
                TokAttr::Control | TokAttr::Unknown => {
                    if special {
                        bytes.extend_from_slice(piece.as_bytes());
                    }
                    // else: render nothing
                }
                TokAttr::Other => {}
            }
        }
        bytes
    }

    fn piece_to_bytes(&self, piece: &str, out: &mut Vec<u8>) {
        for c in piece.chars() {
            match unicode::unicode_to_byte(c) {
                Some(b) => out.push(b),
                None => {
                    // not in the byte map — emit the char's utf-8 bytes verbatim.
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                }
            }
        }
    }

    /// Apply the chat template (from GGUF, or a chatml fallback) to a list of
    /// (role, content) turns, producing the prompt string. Then `encode` it.
    pub fn apply_chat_template(
        &self,
        messages: &[(&str, &str)],
        add_generation_prompt: bool,
    ) -> String {
        chat::apply_chat_template_enc(
            self.chat_template.as_deref(),
            messages,
            add_generation_prompt,
            self.dsv4_encoding,
        )
        // the only Err arm is the dsv4 effort/tool validation, unreachable on this
        // plain-messages path (Default think, no effort, no tools)
        .expect("plain chat render cannot fail")
    }

    /// Does this tokenizer's chat template carry the Qwen3.8 reasoning-effort ladder
    /// (`chat::template_has_qwen_effort`)? Load-bearing for the serve path's plain-render
    /// fast-path decision: on a ladder template the UNSET case renders the vendor's own
    /// `xhigh` default (docs/SERVING.md, reasoning-schema lane 2026-08-23), and only the
    /// tools-capable renderer injects it — `apply_chat_template` reproduces the historical
    /// no-instruction bytes, which on this template are the accepted-and-ignored defect
    /// that lane removed, not a behaviour to preserve.
    pub fn has_qwen_effort_ladder(&self) -> bool {
        self.chat_template
            .as_deref()
            .is_some_and(chat::template_has_qwen_effort)
    }

    /// Tools-capable chat rendering (OpenAI `tools` / `tool_calls` / role:"tool" surface +
    /// the think-tail switch + the per-dialect `reasoning_effort` string). Plain requests
    /// render byte-identically to `apply_chat_template`; see `chat::apply_chat_template_tools`.
    /// The dsv4 encoding revision this tokenizer detected at load rides along, so a dsv4
    /// effort request renders the correct ladder for THIS artifact.
    pub fn apply_chat_template_tools(
        &self,
        turns: &[chat::Turn],
        add_generation_prompt: bool,
        tools_json: &[String],
        think: chat::ThinkMode,
        reasoning_effort: Option<&str>,
    ) -> Result<String, String> {
        chat::apply_chat_template_tools_ex(
            self.chat_template.as_deref(),
            turns,
            add_generation_prompt,
            tools_json,
            &[],
            think,
            reasoning_effort,
            self.dsv4_encoding,
        )
    }

    /// `apply_chat_template_tools` plus the gemma4 arm's structured tool `function` objects
    /// (`tools_struct`). The serve path uses this so gemma4 tool DEFINITIONS render into the
    /// tooluse dialect; every non-gemma dialect ignores `tools_struct`. The dsv4 encoding
    /// revision rides from the tokenizer (see `dsv4_encoding`).
    #[allow(clippy::too_many_arguments)]
    pub fn apply_chat_template_tools_ex(
        &self,
        turns: &[chat::Turn],
        add_generation_prompt: bool,
        tools_json: &[String],
        tools_struct: &[chat::Val],
        think: chat::ThinkMode,
        reasoning_effort: Option<&str>,
    ) -> Result<String, String> {
        chat::apply_chat_template_tools_ex(
            self.chat_template.as_deref(),
            turns,
            add_generation_prompt,
            tools_json,
            tools_struct,
            think,
            reasoning_effort,
            self.dsv4_encoding,
        )
    }
}

enum Fragment {
    Text(String),
    Token(u32),
}

/// deepseek-v4 encoding-revision census over the checkpoint's config.json (0731 re-gate,
/// research/dsv4-template-20260818/ENCODING-DIFF.md). The 0731 checkpoint added exactly
/// these four keys in the same revision that remapped the reasoning-effort ladder, and
/// they are the ONLY artifact-level marker (tokenizer.json / tokenizer_config.json /
/// generation_config.json are byte-identical across preview and 0731):
///
///   - `model_type == "deepseek_v4"` with all four present -> `Some(V0731)`
///   - `model_type == "deepseek_v4"` with none present      -> `Some(Preview)`
///   - config.json absent/unparseable, or a different model family -> `None` (unknown;
///     dsv4 renders then refuse the effort levels whose bytes differ across revisions)
///   - a PARTIAL set -> `Err` (a hand-edited/corrupt config; refuse the load rather
///     than guess an effort ladder)
///
/// Detection reads config CONTENT via the json parser — never filenames or template text.
fn dsv4_encoding_from_config(dir: &std::path::Path) -> Result<Option<chat::Dsv4Encoding>, String> {
    const DSPARK_KEYS: [&str; 4] = [
        "dspark_block_size",
        "dspark_markov_rank",
        "dspark_noise_token_id",
        "dspark_target_layer_ids",
    ];
    let cfg_path = dir.join("config.json");
    let Ok(text) = std::fs::read_to_string(&cfg_path) else {
        return Ok(None);
    };
    let Ok(cfg) = json::parse(&text) else {
        // A config.json the model loader cannot read either; the tokenizer stays honest
        // with "unknown" instead of failing a load the loader may report better.
        return Ok(None);
    };
    if cfg.get("model_type").and_then(|v| v.as_str()) != Some("deepseek_v4") {
        // Not this family's config: no encoding claim (a dsv4 TEMPLATE over a foreign
        // config is a franken artifact — effort-differing renders refuse).
        return Ok(None);
    }
    let present: Vec<&str> = DSPARK_KEYS
        .iter()
        .copied()
        .filter(|k| cfg.get(k).is_some())
        .collect();
    match present.len() {
        0 => Ok(Some(chat::Dsv4Encoding::Preview)),
        4 => Ok(Some(chat::Dsv4Encoding::V0731)),
        _ => Err(format!(
            "{}: partial dspark_* key set {:?} (expected none or all of {:?}) — cannot \
             determine the deepseek-v4 encoding revision; refusing rather than guessing \
             the reasoning-effort ladder",
            cfg_path.display(),
            present,
            DSPARK_KEYS
        )),
    }
}

/// True when an HF `pre_tokenizer` object is byte-level BPE: type == "ByteLevel", or a
/// "Sequence" whose pretokenizers include a ByteLevel step (the common Split+ByteLevel combo).
/// Collect the regexes of every `Split` step in an HF `pre_tokenizer`, in serialization order.
/// A `Sequence` is walked depth-first; non-`Split` steps (ByteLevel, Digits, …) contribute
/// nothing. `{"pattern": {"String": …}}` is not a regex and is skipped.
fn collect_split_regexes(pt: &json::Value, out: &mut Vec<String>) {
    match pt.get("type").and_then(|v| v.as_str()) {
        Some("Sequence") => {
            if let Some(arr) = pt.get("pretokenizers").and_then(|v| v.as_arr()) {
                for step in arr {
                    collect_split_regexes(step, out);
                }
            }
        }
        Some("Split") => {
            if let Some(r) = pt
                .get("pattern")
                .and_then(|p| p.get("Regex"))
                .and_then(|v| v.as_str())
            {
                out.push(r.to_string());
            }
        }
        _ => {}
    }
}

/// Map an ordered set of pre-tokenizer split regexes onto a `tokenizer.ggml.pre` id.
/// Byte-exact comparison against the shipped constants — a near-match is a different splitter
/// (qwen2 vs qwen35 differ by two character classes and produce different ids on marks), so
/// there is deliberately no fuzzy path. `None` = no known family.
fn pre_from_split_regexes(regexes: &[String]) -> Option<&'static str> {
    match regexes {
        [one] if one == QWEN35_PRETOKENIZE_REGEX => Some("qwen35"),
        [one] if one == QWEN2_PRETOKENIZE_REGEX => Some("qwen2"),
        // GLM-5.3-Flash's `tokenizer.ggml.pre` is `default`; this arm is the ONLY thing that
        // identifies it, and the one-atom `{1,3}` delta from qwen2 is why the comparison has
        // to stay byte-exact.
        [one] if one == GLM4_PRETOKENIZE_REGEX => Some("glm4"),
        [a, b, c]
            if a == DEEPSEEK_V3_SPLIT_REGEXES[0]
                && b == DEEPSEEK_V3_SPLIT_REGEXES[1]
                && c == DEEPSEEK_V3_SPLIT_REGEXES[2] =>
        {
            Some("deepseek-v3")
        }
        _ => None,
    }
}

fn pre_tokenizer_is_byte_level(pt: &json::Value) -> bool {
    match pt.get("type").and_then(|v| v.as_str()) {
        Some("ByteLevel") => true,
        Some("Sequence") => pt
            .get("pretokenizers")
            .and_then(|v| v.as_arr())
            .map(|arr| arr.iter().any(pre_tokenizer_is_byte_level))
            .unwrap_or(false),
        _ => false,
    }
}

#[cfg(test)]
mod pretokenizer_tests {
    use super::*;

    /// Every id on the shipped allowlist still resolves — the regression guard for the flip from
    /// warn-and-fall-through to hard-refuse. `gemma4` is the SPM path and pairs with the gemma4
    /// vocab model; the other three are gpt2-vocab splits.
    #[test]
    fn every_supported_pre_resolves() {
        assert_eq!(
            PreSplit::resolve_with("qwen35", false, false),
            Ok(PreSplit::Qwen35)
        );
        assert_eq!(
            PreSplit::resolve_with("qwen2", false, false),
            Ok(PreSplit::Qwen35)
        );
        assert_eq!(
            PreSplit::resolve_with("deepseek-v3", false, false),
            Ok(PreSplit::DeepseekV3)
        );
        assert_eq!(
            PreSplit::resolve_with("glm4", false, false),
            Ok(PreSplit::Glm4)
        );
        assert_eq!(
            PreSplit::resolve_with("gemma4", true, false),
            Ok(PreSplit::Spm)
        );
        // and the allowlist constant is exactly that set, so the error text cannot drift from
        // what the code accepts
        assert_eq!(
            SUPPORTED_PRETOKENIZERS,
            &["qwen35", "qwen2", "deepseek-v3", "gemma4", "glm4"]
        );
    }

    /// An unknown `pre` is a typed error, not a warning and not a wrong split.
    #[test]
    fn unknown_pre_is_a_typed_error() {
        let err =
            PreSplit::resolve_with("llama4", false, false).expect_err("llama4 has no ported split");
        assert_eq!(
            err,
            UnknownPretokenizer {
                pre: "llama4".into(),
                spm_style: false
            }
        );
        let msg = err.to_string();
        // names the offending value, lists what IS supported, and points at the opt-out
        assert!(msg.contains("'llama4'"), "{msg}");
        for supported in SUPPORTED_PRETOKENIZERS {
            assert!(
                msg.contains(supported),
                "error must list {supported}: {msg}"
            );
        }
        assert!(msg.contains(ALLOW_UNKNOWN_PRETOKENIZER_ENV), "{msg}");
        // and it is a real std::error::Error, so `?` from a loader keeps the type
        let _: &dyn std::error::Error = &err;
    }

    /// A `pre`/vocab-model disagreement is its own fault: an SPM vocab with a gpt2 `pre`, or a
    /// gpt2 vocab claiming the gemma4 SPM pre, must not silently pick one side.
    #[test]
    fn pre_and_vocab_model_must_agree() {
        assert!(PreSplit::resolve_with("qwen35", true, false).is_err());
        assert!(PreSplit::resolve_with("gemma4", false, false).is_err());
        // the historical GGUF/HF sentinels for "no pre declared" are refusals, not qwen35
        assert!(PreSplit::resolve_with("default", false, false).is_err());
        assert!(PreSplit::resolve_with("", false, false).is_err());
    }

    /// The opt-out loads, and it declares itself in the resolved split so a gate can refuse it.
    #[test]
    fn opt_out_loads_with_a_fallback_marker() {
        assert_eq!(
            PreSplit::resolve_with("llama4", false, true),
            Ok(PreSplit::UnknownFallbackQwen35)
        );
        // ... including for an SPM-model disagreement
        assert_eq!(
            PreSplit::resolve_with("qwen35", true, true),
            Ok(PreSplit::UnknownFallbackQwen35)
        );
    }

    /// The env name is the one documented, and only an exact `1` engages it (so a stale
    /// `=0`/`=false` in a launcher does not silently turn wrong ids back on).
    #[test]
    fn opt_out_env_gate() {
        let _env = pretokenizer_env_lock();
        // SAFETY: the lock above makes this thread the only one touching this variable, and
        // the resolve paths every other test uses take the decision as a parameter.
        unsafe { std::env::remove_var(ALLOW_UNKNOWN_PRETOKENIZER_ENV) };
        assert!(!allow_unknown_pretokenizer());
        unsafe { std::env::set_var(ALLOW_UNKNOWN_PRETOKENIZER_ENV, "0") };
        assert!(!allow_unknown_pretokenizer());
        unsafe { std::env::set_var(ALLOW_UNKNOWN_PRETOKENIZER_ENV, "1") };
        assert!(allow_unknown_pretokenizer());
        assert_eq!(
            PreSplit::resolve("llama4", false),
            Ok(PreSplit::UnknownFallbackQwen35)
        );
        unsafe { std::env::remove_var(ALLOW_UNKNOWN_PRETOKENIZER_ENV) };
        assert!(PreSplit::resolve("llama4", false).is_err());
    }

    /// Regex identification is byte-exact and order-sensitive: a near-miss is a DIFFERENT
    /// splitter, and a partial deepseek Sequence is not the deepseek Sequence.
    #[test]
    fn split_regex_identification_is_exact() {
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        assert_eq!(
            pre_from_split_regexes(&s(&[QWEN35_PRETOKENIZE_REGEX])),
            Some("qwen35")
        );
        assert_eq!(
            pre_from_split_regexes(&s(&[QWEN2_PRETOKENIZE_REGEX])),
            Some("qwen2")
        );
        assert_eq!(
            pre_from_split_regexes(&s(&DEEPSEEK_V3_SPLIT_REGEXES)),
            Some("deepseek-v3")
        );
        // GLM-5.3-Flash: `tokenizer.ggml.pre` is `default`, so the tokenizer.json Split regex
        // is the ONLY identifier the checkpoint carries.
        assert_eq!(
            pre_from_split_regexes(&s(&[GLM4_PRETOKENIZE_REGEX])),
            Some("glm4")
        );
        // order matters
        assert_eq!(
            pre_from_split_regexes(&s(&[
                DEEPSEEK_V3_SPLIT_REGEXES[1],
                DEEPSEEK_V3_SPLIT_REGEXES[0],
                DEEPSEEK_V3_SPLIT_REGEXES[2],
            ])),
            None
        );
        // a truncated Sequence is not the family
        assert_eq!(
            pre_from_split_regexes(&s(&[
                DEEPSEEK_V3_SPLIT_REGEXES[0],
                DEEPSEEK_V3_SPLIT_REGEXES[1]
            ])),
            None
        );
        // one character off is a different splitter
        let mut near = QWEN35_PRETOKENIZE_REGEX.to_string();
        near.push('x');
        assert_eq!(pre_from_split_regexes(&s(&[&near])), None);
        assert_eq!(pre_from_split_regexes(&[]), None);
        // qwen2 and qwen35 are NOT the same string (the two-class delta is real)
        assert_ne!(QWEN2_PRETOKENIZE_REGEX, QWEN35_PRETOKENIZE_REGEX);
        // glm4 is qwen2's pattern with ONE atom changed — assert that literally, so nobody can
        // "simplify" the two constants into one and silently route GLM through qwen2's split.
        assert_ne!(GLM4_PRETOKENIZE_REGEX, QWEN2_PRETOKENIZE_REGEX);
        assert_eq!(
            QWEN2_PRETOKENIZE_REGEX.replacen(r"|\p{N}|", r"|\p{N}{1,3}|", 1),
            GLM4_PRETOKENIZE_REGEX,
            "glm4 must differ from qwen2 in exactly the digit-run atom"
        );
        // and dropping the {1,3} back off is qwen2, not glm4
        assert_eq!(
            pre_from_split_regexes(&s(&[&GLM4_PRETOKENIZE_REGEX.replacen(
                r"\p{N}{1,3}",
                r"\p{N}",
                1
            )])),
            Some("qwen2")
        );
    }

    /// `collect_split_regexes` walks a Sequence in order and ignores non-Split steps and
    /// `{"String": …}` patterns (gemma's `Split{String:" "}` is not a regex).
    #[test]
    fn collect_split_regexes_walks_in_order() {
        let src = r#"{"type":"Sequence","pretokenizers":[
            {"type":"Split","pattern":{"Regex":"A"},"behavior":"Isolated"},
            {"type":"Split","pattern":{"String":" "},"behavior":"Isolated"},
            {"type":"Digits","individual_digits":true},
            {"type":"Sequence","pretokenizers":[
                {"type":"Split","pattern":{"Regex":"B"},"behavior":"Isolated"}
            ]},
            {"type":"ByteLevel","add_prefix_space":false}
        ]}"#;
        let v = json::parse(src).unwrap();
        let mut out = Vec::new();
        collect_split_regexes(&v, &mut out);
        assert_eq!(out, vec!["A".to_string(), "B".to_string()]);
    }
}

#[cfg(test)]
mod hf_tests {
    use super::*;

    /// Inline tokenizer.json fixture: byte-level BPE, ~20 tokens incl one special added
    /// token, merges deliberately MIXED between the "a b" string format and the [a, b]
    /// pair format (HF emits both across tokenizers versions).
    ///
    /// The `Split` step carries the REAL qwen35 regex (it was an empty string until
    /// 2026-08-19). That is what a shipped Qwen checkpoint looks like, and it is what lets the
    /// no-`tokenizer_config.json` test below identify a pre-tokenizer at all — the empty-regex
    /// fixture only loaded because an unidentified pre-tokenizer used to fall through silently.
    const TOKENIZER_JSON: &str = r#"{
      "version": "1.0",
      "added_tokens": [
        {"id": 15, "content": "<|end|>", "special": true},
        {"id": 16, "content": "<think>", "special": false}
      ],
      "pre_tokenizer": {
        "type": "Sequence",
        "pretokenizers": [
          {"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated"},
          {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false}
        ]
      },
      "model": {
        "type": "BPE",
        "vocab": {
          "h": 0, "e": 1, "l": 2, "o": 3, "Ġ": 4, "w": 5, "r": 6, "d": 7,
          "he": 8, "ll": 9, "hell": 10, "hello": 11, "Ġw": 12, "or": 13, "!": 14
        },
        "merges": [
          "h e",
          ["l", "l"],
          "he ll",
          ["hell", "o"],
          ["Ġ", "w"],
          "o r"
        ]
      }
    }"#;

    fn write_fixture(
        name: &str,
        tokenizer_config: Option<&str>,
        generation_config: Option<&str>,
        jinja: Option<&str>,
    ) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("memra-tok-hf-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("tokenizer.json"), TOKENIZER_JSON).unwrap();
        if let Some(tc) = tokenizer_config {
            std::fs::write(dir.join("tokenizer_config.json"), tc).unwrap();
        }
        if let Some(gc) = generation_config {
            std::fs::write(dir.join("generation_config.json"), gc).unwrap();
        }
        if let Some(j) = jinja {
            std::fs::write(dir.join("chat_template.jinja"), j).unwrap();
        }
        dir
    }

    /// A GLM-5.3-Flash-shaped checkpoint dir: the shared fixture's BPE with the artifact's
    /// three declared eos tokens spliced into `added_tokens` at their REAL ids, plus the two
    /// real vendor sidecars byte-for-byte. What is under test is which declared ids survive
    /// the load, so the vocab/merges/pre-tokenizer stay the shared fixture's.
    fn write_glm53_fixture() -> std::path::PathBuf {
        let base_added = r#""added_tokens": [
        {"id": 15, "content": "<|end|>", "special": true},
        {"id": 16, "content": "<think>", "special": false}
      ],"#;
        // ids straight off the artifact's own added_tokens table (box, 2026-08-28).
        let glm_added = r#""added_tokens": [
        {"id": 154820, "content": "<|endoftext|>", "special": true},
        {"id": 154826, "content": "<|system|>", "special": true},
        {"id": 154827, "content": "<|user|>", "special": true},
        {"id": 154828, "content": "<|assistant|>", "special": true},
        {"id": 154829, "content": "<|observation|>", "special": true}
      ],"#;
        assert!(
            TOKENIZER_JSON.contains(base_added),
            "fixture added_tokens block moved"
        );
        let tj = TOKENIZER_JSON.replace(base_added, glm_added);
        // DENSITY. `from_hf_dir` refuses a tokenizer.json whose ids are sparse relative to its
        // entry count (`MAX_TOKENIZER_SPARSE_*`, PR #57) — a real defense: `id_to_token` is a
        // DENSE Vec, so a 20-entry file declaring id 154829 would allocate a 154k-slot table
        // from a few hundred bytes of attacker-chosen JSON. The GLM artifact itself is dense
        // (154830 ids, ~154k entries); it is this FIXTURE that was sparse, having borrowed a
        // 17-id vocab and bolted the artifact's real high ids onto it. Fill the gap so the
        // fixture has the artifact's SHAPE and the guard sees what it would see in production.
        // Ids 17..154820 only: the five added_tokens above own 154820..154829, and 0..16 are
        // the shared fixture's own vocab.
        let anchor = "\"!\": 14\n";
        assert!(tj.contains(anchor), "fixture vocab tail moved");
        let mut filler = String::with_capacity(4 << 20);
        filler.push_str("\"!\": 14");
        for id in 17u32..154_820 {
            // A name that can never collide with a real piece, a merge operand, or any of the
            // eos NAMES the name-keyed backstop looks for: no merge rule can ever produce it.
            filler.push_str(&format!(",\n          \"##fill{id}\": {id}"));
        }
        filler.push('\n');
        let tj = tj.replace(anchor, &filler);

        let dir = std::env::temp_dir().join(format!("memra-tok-glm53-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("tokenizer.json"), tj).unwrap();
        std::fs::write(dir.join("tokenizer_config.json"), GLM53_TOKENIZER_CONFIG).unwrap();
        std::fs::write(dir.join("generation_config.json"), GLM53_GENERATION_CONFIG).unwrap();
        dir
    }

    #[test]
    fn hf_dir_encode_decode_roundtrip_and_specials() {
        // eos as an AddedToken OBJECT + chat_template string in tokenizer_config.
        let tc = r#"{
          "eos_token": {"content": "<|end|>", "lstrip": false},
          "add_bos_token": false,
          "pretokenize_regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
          "chat_template": "{{ messages }}<|end|>"
        }"#;
        let dir = write_fixture("full", Some(tc), None, None);
        let tok = Tokenizer::from_hf_dir(&dir).expect("from_hf_dir");

        assert_eq!(tok.eos_id(), 15);
        assert_eq!(tok.bos_id(), None);
        assert_eq!(tok.pre(), "qwen35");
        assert_eq!(tok.vocab_size(), 17); // ids 0..16 (added tokens extend the table)
        assert_eq!(tok.chat_template(), Some("{{ messages }}<|end|>"));

        // BPE over both merge formats: "hello world" -> hello(11) Ġw(12) or(13) l(2) d(7).
        // The 'hello' chain exercises string merges (h e / he ll), the pair merges
        // ([l,l] / [hell,o] / [Ġ,w]) fire inside the same words -> both formats load.
        let ids = tok.encode("hello world", true);
        assert_eq!(ids, vec![11, 12, 13, 2, 7]);
        assert_eq!(tok.decode(&ids), "hello world");

        // special handling: <|end|> (Control) is split out BEFORE BPE and never byte-merged.
        let ids = tok.encode("hello<|end|> world", true);
        assert_eq!(ids, vec![11, 15, 12, 13, 2, 7]);
        // decode with specials rendered vs dropped
        assert_eq!(tok.decode_special(&ids, true), "hello<|end|> world");
        assert_eq!(tok.decode_special(&ids, false), "hello world");

        // non-special added token stays Normal: decodes as literal text.
        assert_eq!(tok.decode(&[16]), "<think>");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn hf_dir_generation_config_eos_fallback_and_jinja() {
        // no tokenizer_config eos -> generation_config eos_token_id (array form) must win;
        // chat template comes from chat_template.jinja.
        let gc = r#"{"eos_token_id": [15, 14]}"#;
        let dir = write_fixture("genconf", None, Some(gc), Some("JINJA {{ messages }}"));
        let tok = Tokenizer::from_hf_dir(&dir).expect("from_hf_dir");
        assert_eq!(tok.eos_id(), 15);
        // DELIBERATE CHANGE, 2026-08-27 (lane glm53-flash-bringup): the scalar still takes
        // the array's first entry — unchanged — but id 14 is no longer dropped. Every id in
        // the vendor's array is a declared stop; keeping only the first is the GLM-5.3 bug.
        assert_eq!(tok.eos_ids(), &[15, 14]);
        assert_eq!(tok.eog_ids(), vec![15, 14]);
        assert!(!tok.encode("hello", true).is_empty());
        assert_eq!(tok.chat_template(), Some("JINJA {{ messages }}"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// GLM-5.3-Flash's REAL banked sidecars, byte-for-byte.
    ///
    ///   research/glm53-flash-bringup-20260827/generation_config.json
    ///     sha256 230c30609ecbbb9e6583bedde8e7bdda0c6eb8fe5fad0eaeb3d1b293d751cb4f
    ///   research/glm53-flash-bringup-20260827/tokenizer_config.json
    ///     sha256 98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc
    ///
    /// Inlined rather than `include_str!`d: these crates publish to crates.io and `research/`
    /// is outside every package, so a path include would compile here and break `cargo
    /// publish`. The lane dir holds the same bytes for provenance.
    const GLM53_GENERATION_CONFIG: &str = r#"{
  "_from_model_config": true,
  "eos_token_id": [
    154820,
    154827,
    154829
  ],
  "pad_token_id": 154820,
  "temperature": 1.0,
  "top_p": 0.95,
  "transformers_version": "5.16.0"
}
"#;
    const GLM53_TOKENIZER_CONFIG: &str = r#"{
  "backend": "tokenizers",
  "clean_up_tokenization_spaces": false,
  "do_lower_case": false,
  "eos_token": "<|endoftext|>",
  "extra_special_tokens": ["<|endoftext|>", "[MASK]", "[gMASK]", "[sMASK]", "<sop>", "<eop>",
    "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>", "<|begin_of_image|>",
    "<|end_of_image|>", "<|begin_of_video|>", "<|end_of_video|>", "<|begin_of_audio|>",
    "<|end_of_audio|>", "<|begin_of_transcription|>", "<|end_of_transcription|>"],
  "is_local": true,
  "model_max_length": 1048576,
  "pad_token": "<|endoftext|>",
  "padding_side": "left",
  "remove_space": false,
  "tokenizer_class": "TokenizersBackend"
}"#;

    /// THE GLM-5.3-FLASH TRIPLE-EOS PIN (lane glm53-flash-bringup, 2026-08-27).
    ///
    /// What was broken: `generation_config.json` declares three eos ids and the engine carried
    /// one, so a finished answer ran straight through `<|user|>` into a hallucinated
    /// multi-turn transcript and the OpenAI surface reported `finish_reason: "length"`
    /// (receipts: research/glm53-flash-bringup-20260827/forward-bisect-receipts/).
    ///
    /// Why <|user|> and <|observation|> are legitimately stops and not arbitrary specials:
    /// they are the artifact's TURN BOUNDARIES. chat_template.jinja emits `<|user|>` as the
    /// user-turn prefix (line 139) and `<|observation|>` as the tool-result prefix (line 167),
    /// with `<|assistant|>` opening the generation the server asked for (line 256); an
    /// assistant that emits either has left its own turn. Both are `special: true` added
    /// tokens in the artifact's tokenizer.json (ids 154827 / 154829, verified on the box
    /// 2026-08-28), i.e. control tokens that can never fall out of ordinary BPE text.
    ///
    /// The ids come from the vendor sidecar, so the fixture's `tokenizer.json` only has to
    /// carry them; the pre-tokenizer here is the shared qwen35 fixture and is not under test.
    #[test]
    fn hf_dir_glm53_flash_carries_all_three_declared_eos_ids() {
        let dir = write_glm53_fixture();
        let tok = Tokenizer::from_hf_dir(&dir).expect("from_hf_dir");

        // The SCALAR is unchanged by this lane: tokenizer_config's eos_token still wins, and
        // it is <|endoftext|> — the boot log, embed pooling and the grammar bridges all keep
        // seeing exactly what they saw before.
        assert_eq!(tok.eos_id(), 154_820);
        assert_eq!(tok.id_of("<|endoftext|>"), Some(154_820));
        assert_eq!(tok.id_of("<|user|>"), Some(154_827));
        assert_eq!(tok.id_of("<|observation|>"), Some(154_829));

        // The SET is what the serve path stops on (`worker::run` unions `eog_ids()` into
        // `GenParams::eos`). Set EQUALITY, not containment: this vocab has no <|im_end|> /
        // <turn|> / <end_of_turn>, so the name-keyed backstop must contribute nothing and the
        // three declared ids must be all of it.
        assert_eq!(tok.eos_ids(), &[154_820, 154_827, 154_829]);
        assert_eq!(tok.eog_ids(), vec![154_820, 154_827, 154_829]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// REGRESSION PIN: a single-eos family does not move.
    ///
    /// Both single-id shapes that ship today — `tokenizer_config.eos_token` with no
    /// generation_config at all, and hy3's scalar `generation_config.eos_token_id` — must
    /// produce a one-id `eos_ids` and an `eog_ids` identical to the pre-lane
    /// `vec![eos_id] + name-keyed backstop`.
    #[test]
    fn hf_dir_single_eos_family_stop_set_unchanged() {
        // shape 1: tokenizer_config eos_token only (no generation_config).
        let tc = r#"{"eos_token": "<|end|>", "add_bos_token": false}"#;
        let dir = write_fixture("single-tc", Some(tc), None, None);
        let tok = Tokenizer::from_hf_dir(&dir).expect("from_hf_dir");
        assert_eq!(tok.eos_id(), 15);
        assert_eq!(tok.eos_ids(), &[15]);
        assert_eq!(tok.eog_ids(), vec![15]);
        let _ = std::fs::remove_dir_all(&dir);

        // shape 2: hy3 — tokenizer_config eos_token AND a scalar generation_config id that
        // agrees. One id in, one id out; the scalar arm must not become a one-element array
        // that then re-enters as a duplicate.
        let dir = write_fixture("single-gc", Some(tc), Some(r#"{"eos_token_id": 15}"#), None);
        let tok = Tokenizer::from_hf_dir(&dir).expect("from_hf_dir");
        assert_eq!(tok.eos_id(), 15);
        assert_eq!(tok.eos_ids(), &[15]);
        assert_eq!(tok.eog_ids(), vec![15]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// THE CENSUS DELTA, made explicit rather than discovered in production.
    ///
    /// Reading the whole array changes the stop set of every from_hf_dir family whose vendor
    /// declares more than one id. Measured 2026-08-27 over the artifacts on the rig:
    ///
    ///   qwen36/qwen38 27B  eos_token_id [248046, 248044]  -> +248044 `<|endoftext|>`
    ///                      (248046 `<|im_end|>` was already in via the name backstop)
    ///   qwen35 9B          scalar 248044, tc `<|im_end|>` -> +248044 `<|endoftext|>`
    ///   qwen3 1.7B         [151645, 151643]               -> +151643 `<|endoftext|>`
    ///   gemma4 26B NVFP4   [1, 106, 50]                   -> +50 `<|tool_response>`
    ///                      (1 `<eos>` is the scalar, 106 `<turn|>` was in via the backstop)
    ///   hy3                scalar 120025                  -> no change
    ///   m3                 scalar 200020                  -> no change
    ///   step35 NVFP4       [1, 2, 128007]                 -> +1 end-of-sentence, +2 pad
    ///                      (128007 `<|im_end|>` is the scalar; the vendor lists its PAD
    ///                       token as an eos — a control token either way)
    ///   every GGUF family                                 -> no change by construction
    ///   dsv4                                              -> no change (dsv4_serve.rs builds
    ///                                                        its stop set from `eos_id()`)
    ///
    /// Every delta is a `special: true` control token the vendor itself names as a stop, and
    /// none can be produced by ordinary BPE over text. This pin holds the qwen shape so the
    /// delta is a decision with a test behind it, not a surprise.
    #[test]
    fn hf_dir_multi_eos_array_extends_the_stop_set_beyond_the_first_id() {
        // qwen shape: tokenizer_config names the turn-end token, generation_config's array
        // adds the raw end-of-text id after it.
        let tc = r#"{"eos_token": "<|end|>", "add_bos_token": false}"#;
        let gc = r#"{"eos_token_id": [15, 14]}"#;
        let dir = write_fixture("multi", Some(tc), Some(gc), None);
        let tok = Tokenizer::from_hf_dir(&dir).expect("from_hf_dir");
        assert_eq!(
            tok.eos_id(),
            15,
            "the scalar keeps tokenizer_config's choice"
        );
        assert_eq!(tok.eos_ids(), &[15, 14]);
        assert_eq!(tok.eog_ids(), vec![15, 14]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The three-Split deepseek-v3 pre-tokenizer Sequence, byte-for-byte as HF serializes it
    /// (Hy3 / Step-3.7-Flash). This is the case that used to land on `default` -> the silent
    /// qwen35 fallback even though `unicode::split_deepseek_v3` already existed.
    #[test]
    fn hf_dir_identifies_deepseek_v3_from_tokenizer_json() {
        let dsv3_pt = r##""pre_tokenizer": {
        "type": "Sequence",
        "pretokenizers": [
          {"type": "Split", "pattern": {"Regex": "\\p{N}{1,3}"}, "behavior": "Isolated"},
          {"type": "Split", "pattern": {"Regex": "[一-龥぀-ゟ゠-ヿ]+"}, "behavior": "Isolated"},
          {"type": "Split", "pattern": {"Regex": "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated"},
          {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}
        ]
      },"##;
        // splice the deepseek pre_tokenizer into the shared fixture in place of the qwen one
        let open = TOKENIZER_JSON.find(r#""pre_tokenizer""#).unwrap();
        let close = TOKENIZER_JSON.find(r#""model""#).unwrap();
        let json = format!(
            "{}{}\n      {}",
            &TOKENIZER_JSON[..open],
            dsv3_pt,
            &TOKENIZER_JSON[close..]
        );
        let dir = std::env::temp_dir().join(format!("memra-tok-hf-dsv3-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("tokenizer.json"), &json).unwrap();
        std::fs::write(
            dir.join("generation_config.json"),
            r#"{"eos_token_id": 15}"#,
        )
        .unwrap();
        let tok = Tokenizer::from_hf_dir(&dir).expect("from_hf_dir");
        assert_eq!(tok.pre(), "deepseek-v3");
        assert_eq!(tok.split(), PreSplit::DeepseekV3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The `qwen2` regex differs from qwen35 by two character classes and must be identified as
    /// qwen2, not silently mistaken for qwen35 (they share a state machine but not an id).
    #[test]
    fn hf_dir_identifies_qwen2_regex() {
        let json = TOKENIZER_JSON.replace(
            r"[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+",
            r"[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+",
        );
        assert_ne!(json, TOKENIZER_JSON, "the qwen2 substitution must apply");
        let dir = std::env::temp_dir().join(format!("memra-tok-hf-qwen2-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("tokenizer.json"), &json).unwrap();
        std::fs::write(
            dir.join("generation_config.json"),
            r#"{"eos_token_id": 15}"#,
        )
        .unwrap();
        let tok = Tokenizer::from_hf_dir(&dir).expect("from_hf_dir");
        assert_eq!(tok.pre(), "qwen2");
        assert_eq!(
            tok.split(),
            PreSplit::Qwen35,
            "qwen2 rides the qwen35 split"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// GLM-5.3-Flash's own shape: `tokenizer.ggml.pre` is absent (the GGUF sentinel is
    /// `default`), the checkpoint carries no `pretokenize_regex` sidecar, and the
    /// tokenizer.json Split regex is the ONLY identifier. It differs from qwen2 by the single
    /// `\p{N}` -> `\p{N}{1,3}` atom, so the byte-exact comparison is the whole gate: a fuzzy
    /// match here would route GLM through qwen2's split and every id downstream would be wrong.
    #[test]
    fn hf_dir_identifies_glm4_regex() {
        let json = TOKENIZER_JSON.replace(
            r"[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+",
            r"[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+",
        );
        assert_ne!(json, TOKENIZER_JSON, "the glm4 substitution must apply");
        let dir = std::env::temp_dir().join(format!("memra-tok-hf-glm4-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("tokenizer.json"), &json).unwrap();
        std::fs::write(
            dir.join("generation_config.json"),
            r#"{"eos_token_id": 15}"#,
        )
        .unwrap();
        let tok = Tokenizer::from_hf_dir(&dir).expect("from_hf_dir");
        assert_eq!(tok.pre(), "glm4");
        assert_eq!(
            tok.split(),
            PreSplit::Glm4,
            "glm4 gets its OWN split, not qwen35's"
        );
        // and the SAME fixture with qwen2's `\p{N}` instead resolves elsewhere — the one-atom
        // delta is the whole identification, so both directions are pinned here.
        let qjson = TOKENIZER_JSON.replace(
            r"[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+",
            r"[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+",
        );
        let qwen_dir =
            std::env::temp_dir().join(format!("memra-tok-hf-glm4-ctl-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&qwen_dir);
        std::fs::create_dir_all(&qwen_dir).unwrap();
        std::fs::write(qwen_dir.join("tokenizer.json"), &qjson).unwrap();
        std::fs::write(
            qwen_dir.join("generation_config.json"),
            r#"{"eos_token_id": 15}"#,
        )
        .unwrap();
        let qtok = Tokenizer::from_hf_dir(&qwen_dir).expect("from_hf_dir(qwen2 control)");
        assert_eq!(qtok.pre(), "qwen2");
        assert_eq!(qtok.split(), PreSplit::Qwen35);
        assert_ne!(tok.split(), qtok.split());
        // the two splits disagree on a 4-digit run — the reason the variant exists
        assert_eq!(unicode::split_glm4("1234"), ["123", "4"]);
        assert_eq!(unicode::split_qwen35("1234"), ["1", "2", "3", "4"]);
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&qwen_dir);
    }

    /// An HF checkpoint whose pre-tokenizer matches nothing known is REFUSED, and the error
    /// names both observations so the next porter knows what to implement.
    #[test]
    fn hf_dir_refuses_unidentifiable_pretokenizer() {
        // This test's whole point is the REFUSAL path, which the opt-out env var disables, so
        // it holds the same lock as `opt_out_env_gate` and clears the variable first: a panic
        // partway through that test can otherwise leave the opt-out set.
        let _env = pretokenizer_env_lock();
        // SAFETY: the lock above makes this thread the only one touching this variable.
        unsafe { std::env::remove_var(ALLOW_UNKNOWN_PRETOKENIZER_ENV) };
        let json = TOKENIZER_JSON.replace(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|", "SOMETHING-ELSE|");
        assert_ne!(json, TOKENIZER_JSON);
        let dir = std::env::temp_dir().join(format!("memra-tok-hf-unk-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("tokenizer.json"), &json).unwrap();
        std::fs::write(
            dir.join("generation_config.json"),
            r#"{"eos_token_id": 15}"#,
        )
        .unwrap();
        let err = match Tokenizer::from_hf_dir(&dir) {
            Ok(_) => panic!("unidentifiable pre must refuse to load"),
            Err(e) => e,
        };
        assert!(
            err.contains("unsupported tokenizer.ggml.pre 'default'"),
            "{err}"
        );
        assert!(
            err.contains("SOMETHING-ELSE"),
            "error must quote the regex: {err}"
        );
        assert!(err.contains("MEMRA_ALLOW_UNKNOWN_PRETOKENIZER"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Artifact-backed: real vendor `tokenizer.json` files must resolve to the right split.
    /// Per-entry skip when a checkpoint is not staged (same posture as tests/llama_parity.rs) —
    /// this is the gate that pins the regex constants against real vendor serializations rather
    /// than against our own fixture. Every one of these landed on the silent qwen35 fallback
    /// before 2026-08-19 except the Qwen entry (which carries `pretokenize_regex`).
    #[test]
    fn staged_checkpoints_resolve_their_own_pretokenizer() {
        let cases: &[(&str, &str)] = &[
            // ships the deepseek-v3 Sequence verbatim; was mis-tokenized as qwen35
            (
                "/data/ai-ml/hf-models/hy3-layer103p5-sparse-source",
                "deepseek-v3",
            ),
            // qwen2 regex in tokenizer.json, no `pretokenize_regex` sidecar
            ("/data/ai-ml/hf-models/qwen3-1.7b-blk128fp8-synth", "qwen2"),
            // the control: `pretokenize_regex` present and byte-equal
            ("/data/ai-ml/hf-models/qwen35-9b-hf", "qwen35"),
        ];
        let mut ran = 0;
        for (path, want) in cases {
            let dir = std::path::Path::new(path);
            if !dir.join("tokenizer.json").exists() {
                eprintln!("skip: {path} not staged");
                continue;
            }
            let tok = Tokenizer::from_hf_dir(dir).unwrap_or_else(|e| panic!("{path}: {e}"));
            assert_eq!(tok.pre(), *want, "{path}");
            ran += 1;
        }
        eprintln!("staged_checkpoints_resolve_their_own_pretokenizer: {ran}/3 cases ran");
    }

    #[test]
    fn hf_dir_rejects_non_byte_level() {
        let dir = std::env::temp_dir().join(format!("memra-tok-hf-nonbl-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let bad = TOKENIZER_JSON.replace("\"ByteLevel\"", "\"Metaspace\"");
        std::fs::write(dir.join("tokenizer.json"), bad).unwrap();
        assert!(Tokenizer::from_hf_dir(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// deepseek-v4 encoding-revision detection from config.json (0731 re-gate,
    /// ENCODING-DIFF.md): the dspark_* key census — never a filename — decides the ladder.
    #[test]
    fn hf_dir_dsv4_encoding_detection() {
        let gc = r#"{"eos_token_id": [15]}"#;
        let full_dspark = r#""dspark_block_size": 5, "dspark_markov_rank": 256,
             "dspark_noise_token_id": 128799, "dspark_target_layer_ids": [40, 41, 42]"#;

        // no config.json -> unknown (tokenizer-only ref dirs).
        let dir = write_fixture("dsv4-none", None, Some(gc), None);
        let tok = Tokenizer::from_hf_dir(&dir).unwrap();
        assert_eq!(tok.dsv4_encoding(), None);
        let _ = std::fs::remove_dir_all(&dir);

        // deepseek_v4 config without dspark keys -> Preview.
        let dir = write_fixture("dsv4-preview", None, Some(gc), None);
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type": "deepseek_v4", "num_hidden_layers": 43}"#,
        )
        .unwrap();
        let tok = Tokenizer::from_hf_dir(&dir).unwrap();
        assert_eq!(tok.dsv4_encoding(), Some(chat::Dsv4Encoding::Preview));
        let _ = std::fs::remove_dir_all(&dir);

        // deepseek_v4 config with ALL FOUR dspark keys -> V0731.
        let dir = write_fixture("dsv4-0731", None, Some(gc), None);
        std::fs::write(
            dir.join("config.json"),
            format!(r#"{{"model_type": "deepseek_v4", {full_dspark}}}"#),
        )
        .unwrap();
        let tok = Tokenizer::from_hf_dir(&dir).unwrap();
        assert_eq!(tok.dsv4_encoding(), Some(chat::Dsv4Encoding::V0731));
        let _ = std::fs::remove_dir_all(&dir);

        // a PARTIAL dspark key set is ambiguous -> the load refuses.
        let dir = write_fixture("dsv4-partial", None, Some(gc), None);
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type": "deepseek_v4", "dspark_block_size": 5}"#,
        )
        .unwrap();
        let err = match Tokenizer::from_hf_dir(&dir) {
            Err(e) => e,
            Ok(_) => panic!("a partial dspark_* config must refuse the load"),
        };
        assert!(err.contains("partial dspark_*"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);

        // another family's config makes no dsv4 encoding claim -> unknown.
        let dir = write_fixture("dsv4-foreign", None, Some(gc), None);
        std::fs::write(dir.join("config.json"), r#"{"model_type": "qwen3"}"#).unwrap();
        let tok = Tokenizer::from_hf_dir(&dir).unwrap();
        assert_eq!(tok.dsv4_encoding(), None);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
