//! Unicode helpers ported 1:1 from llama.cpp `src/unicode.cpp` / `src/unicode.h`.
//!
//! Only the pieces the GPT-2/qwen35/deepseek-v3 BPE paths need are ported:
//!   - codepoint flags (`\p{L}`, `\p{N}`, `\p{M}`, `\p{P}`, `\p{S}`, `\s`) via the generated
//!     range table, plus `category_flag()` for the collapsed-text map
//!   - `unicode_tolower` (binary search over the lowercase map)
//!   - the GPT-2 byte<->unicode map (`bytes_to_unicode`)
//!   - the hand-written `unicode_regex_split_custom_qwen35` pre-tokenizer state machine
//!   - `split_deepseek_v3`: an equivalent for `LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM`, which
//!     upstream runs through generic `std::regex` (three ordered passes over collapsed text)
//!     rather than a custom splitter. Required by Step-3.5/3.7-Flash.
//!
//! Keeping the classification table identical to llama.cpp is what makes the
//! pre-tokenizer split — and therefore the final token ids — integer-exact.

use crate::unicode_data::{UNICODE_MAP_LOWERCASE, UNICODE_RANGES_FLAGS, UNICODE_SET_WHITESPACE};
use std::collections::HashMap;
use std::sync::OnceLock;

const MAX_CODEPOINTS: usize = 0x110000;

// flag bits (llama.cpp `unicode_cpt_flags` enum). A few are unused by the qwen35
// split but kept for completeness / documentation of the table layout.
#[allow(dead_code)]
mod flag {
    pub const UNDEFINED: u16 = 0x0001;
    pub const NUMBER: u16 = 0x0002; // \p{N}
    pub const LETTER: u16 = 0x0004; // \p{L}
    pub const SEPARATOR: u16 = 0x0008; // \p{Z}
    pub const ACCENT_MARK: u16 = 0x0010; // \p{M}
    pub const PUNCTUATION: u16 = 0x0020; // \p{P}
    pub const SYMBOL: u16 = 0x0040; // \p{S}
    pub const CONTROL: u16 = 0x0080; // \p{C}
    pub const WHITESPACE: u16 = 0x0100; // \s
}
pub use flag::{
    ACCENT_MARK as FLAG_ACCENT_MARK, LETTER as FLAG_LETTER, NUMBER as FLAG_NUMBER,
    UNDEFINED as FLAG_UNDEFINED, WHITESPACE as FLAG_WHITESPACE,
};

/// Codepoint classification flags, mirroring `unicode_cpt_flags`.
/// We only carry the bits the qwen35 pre-tokenizer reads.
#[derive(Clone, Copy, Default)]
pub struct CptFlags(pub u16);

impl CptFlags {
    #[inline]
    pub fn is_number(self) -> bool {
        self.0 & FLAG_NUMBER != 0
    }
    #[inline]
    pub fn is_letter(self) -> bool {
        self.0 & FLAG_LETTER != 0
    }
    #[inline]
    pub fn is_accent_mark(self) -> bool {
        self.0 & FLAG_ACCENT_MARK != 0
    }
    #[inline]
    pub fn is_whitespace(self) -> bool {
        self.0 & FLAG_WHITESPACE != 0
    }
    /// matches `unicode_cpt_flags::as_uint()` for the bits we keep — used by the
    /// qwen35 split to test "any defined category at all".
    #[inline]
    pub fn as_uint(self) -> u16 {
        self.0
    }
    /// `unicode_cpt_flags::category_flag()` — `as_uint() & MASK_CATEGORIES`. Note this is the
    /// whole low byte, not a single bit: a codepoint carrying two category bits yields a value
    /// absent from llama.cpp's `k_ucat_cpt` map and therefore collapses to the 0xD0 fallback.
    #[inline]
    pub fn category_flag(self) -> u16 {
        self.0 & 0x00FF
    }
}

/// Build the full codepoint->flags table, exactly like `unicode_cpt_flags_array()`.
fn cpt_flags_table() -> &'static Vec<u16> {
    static TABLE: OnceLock<Vec<u16>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut flags = vec![FLAG_UNDEFINED; MAX_CODEPOINTS];
        // ranges: [start_i, start_{i+1}) gets range_i.flags
        for i in 1..UNICODE_RANGES_FLAGS.len() {
            let (ini, fl) = UNICODE_RANGES_FLAGS[i - 1];
            let (end, _) = UNICODE_RANGES_FLAGS[i];
            for cpt in ini..end {
                flags[cpt as usize] = fl;
            }
        }
        // whitespace OR-in (note: this OR matches llama's `is_whitespace = true`)
        for &cpt in UNICODE_SET_WHITESPACE.iter() {
            flags[cpt as usize] |= FLAG_WHITESPACE;
        }
        // (lowercase/uppercase/nfd bits are unused by the qwen35 pre-tokenizer)
        flags
    })
}

/// `unicode_cpt_flags_from_cpt` — out-of-range cpts get UNDEFINED (0x0001), matching llama.
#[inline]
pub fn cpt_flags_from_cpt(cpt: u32) -> CptFlags {
    let table = cpt_flags_table();
    if (cpt as usize) < MAX_CODEPOINTS {
        CptFlags(table[cpt as usize])
    } else {
        CptFlags(FLAG_UNDEFINED)
    }
}

/// `unicode_tolower` — binary search over the lowercase map, identity if absent.
#[inline]
pub fn tolower(cpt: u32) -> u32 {
    match UNICODE_MAP_LOWERCASE.binary_search_by(|&(k, _)| k.cmp(&cpt)) {
        Ok(idx) => UNICODE_MAP_LOWERCASE[idx].1,
        Err(_) => cpt,
    }
}

// ---- GPT-2 byte <-> unicode map (`bytes_to_unicode`) ----------------------------------

/// (byte -> unicode-codepoint, codepoint -> byte). Mirrors `unicode_byte_to_utf8_map`.
fn byte_unicode_maps() -> &'static (Vec<char>, HashMap<char, u8>) {
    static MAPS: OnceLock<(Vec<char>, HashMap<char, u8>)> = OnceLock::new();
    MAPS.get_or_init(|| {
        // byte -> char, exactly like the C++ map build order.
        let mut byte_to_char: Vec<Option<char>> = vec![None; 256];
        let mut set = |ch: u32| {
            byte_to_char[ch as usize] = Some(char::from_u32(ch).unwrap());
        };
        for ch in 0x21..=0x7E {
            set(ch);
        }
        for ch in 0xA1..=0xAC {
            set(ch);
        }
        for ch in 0xAE..=0xFF {
            set(ch);
        }
        let mut n: u32 = 0;
        for ch in 0..256u32 {
            if byte_to_char[ch as usize].is_none() {
                byte_to_char[ch as usize] = Some(char::from_u32(256 + n).unwrap());
                n += 1;
            }
        }
        let b2c: Vec<char> = byte_to_char.into_iter().map(|c| c.unwrap()).collect();
        let mut c2b: HashMap<char, u8> = HashMap::with_capacity(256);
        for (b, &c) in b2c.iter().enumerate() {
            c2b.insert(c, b as u8);
        }
        (b2c, c2b)
    })
}

/// Map one raw byte to its GPT-2 unicode char (`unicode_byte_to_utf8`).
#[inline]
pub fn byte_to_unicode(byte: u8) -> char {
    byte_unicode_maps().0[byte as usize]
}

/// Map a GPT-2 unicode char back to its raw byte (`unicode_utf8_to_byte`); None if not in map.
#[inline]
pub fn unicode_to_byte(c: char) -> Option<u8> {
    byte_unicode_maps().1.get(&c).copied()
}

/// GPT-2 byte-encode a raw &str: each *byte* becomes one unicode char.
/// Mirrors `unicode_byte_encoding_process` (which encodes per-byte, not per-cpt).
pub fn byte_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for &b in s.as_bytes() {
        out.push(byte_to_unicode(b));
    }
    out
}

// ---- qwen35 pre-tokenizer split -------------------------------------------------------

/// Port of `unicode_regex_split_custom_qwen35` (llama.cpp `src/unicode.cpp`).
///
/// Splits `text` (a UTF-8 string) into pre-token word boundaries, returning the
/// byte slices for each word. This is a deterministic codepoint-class state machine,
/// NOT a regex engine — that is exactly why it can be ported integer-exact.
///
/// The qwen35 regex it implements:
///   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ | \p{N}
///     | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
pub fn split_qwen35(text: &str) -> Vec<String> {
    // codepoints + the byte length of each, so we can recover substrings.
    let cpts: Vec<u32> = text.chars().map(|c| c as u32).collect();
    let cpt_bytes: Vec<usize> = text.chars().map(|c| c.len_utf8()).collect();
    let n = cpts.len();

    const OOR: u32 = 0xFFFF_FFFF;
    let get_cpt = |pos: usize| -> u32 { if pos < n { cpts[pos] } else { OOR } };
    let get_flags = |pos: usize| -> CptFlags {
        if pos < n {
            cpt_flags_from_cpt(cpts[pos])
        } else {
            CptFlags::default()
        }
    };

    // emit token boundaries as codepoint counts, then convert to byte substrings.
    let mut lens: Vec<usize> = Vec::new(); // codepoint-length of each word
    let mut prev_end = 0usize;
    let add_token = |end: usize, prev_end: &mut usize, lens: &mut Vec<usize>| -> usize {
        debug_assert!(*prev_end <= end && end <= n);
        let len = end - *prev_end;
        if len > 0 {
            lens.push(len);
        }
        *prev_end = end;
        len
    };

    let mut pos = 0usize;
    while pos < n {
        let cpt = get_cpt(pos);
        let flags = get_flags(pos);

        // regex: (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if cpt == b'\'' as u32 && pos + 1 < n {
            let cpt_next = tolower(get_cpt(pos + 1));
            if cpt_next == 's' as u32
                || cpt_next == 't' as u32
                || cpt_next == 'm' as u32
                || cpt_next == 'd' as u32
            {
                pos += add_token(pos + 2, &mut prev_end, &mut lens);
                continue;
            }
            if pos + 2 < n {
                let cpt_nn = tolower(get_cpt(pos + 2));
                if (cpt_next == 'r' as u32 && cpt_nn == 'e' as u32)
                    || (cpt_next == 'v' as u32 && cpt_nn == 'e' as u32)
                    || (cpt_next == 'l' as u32 && cpt_nn == 'l' as u32)
                {
                    pos += add_token(pos + 3, &mut prev_end, &mut lens);
                    continue;
                }
            }
        }

        // regex: [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
        if !(cpt == '\r' as u32 || cpt == '\n' as u32 || flags.is_number())
            && (flags.is_letter()
                || flags.is_accent_mark()
                || get_flags(pos + 1).is_accent_mark()
                || get_flags(pos + 1).is_letter())
        {
            pos += 1;
            while get_flags(pos).is_letter() || get_flags(pos).is_accent_mark() {
                pos += 1;
            }
            add_token(pos, &mut prev_end, &mut lens);
            continue;
        }

        // regex: \p{N}
        if flags.is_number() {
            pos += 1;
            add_token(pos, &mut prev_end, &mut lens);
            continue;
        }

        // regex: <space>?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
        let mut flags2 = if cpt == ' ' as u32 {
            get_flags(pos + 1)
        } else {
            flags
        };
        if !(flags2.is_whitespace()
            || flags2.is_letter()
            || flags2.is_accent_mark()
            || flags2.is_number())
            && flags.as_uint() != 0
        {
            pos += (cpt == ' ' as u32) as usize;
            while !(flags2.is_whitespace()
                || flags2.is_letter()
                || flags2.is_accent_mark()
                || flags2.is_number())
                && flags2.as_uint() != 0
            {
                pos += 1;
                flags2 = get_flags(pos);
            }
            let mut cpt2 = get_cpt(pos);
            while cpt2 == '\r' as u32 || cpt2 == '\n' as u32 {
                pos += 1;
                cpt2 = get_cpt(pos);
            }
            add_token(pos, &mut prev_end, &mut lens);
            continue;
        }

        // count run of whitespace, remember last \r/\n end
        let mut num_ws = 0usize;
        let mut last_end_rn = 0usize;
        while get_flags(pos + num_ws).is_whitespace() {
            let cpt2 = get_cpt(pos + num_ws);
            if cpt2 == '\r' as u32 || cpt2 == '\n' as u32 {
                last_end_rn = pos + num_ws + 1;
            }
            num_ws += 1;
        }

        // regex: \s*[\r\n]+
        if last_end_rn > 0 {
            pos = last_end_rn;
            add_token(pos, &mut prev_end, &mut lens);
            continue;
        }

        // regex: \s+(?!\S)
        if num_ws > 1 && get_cpt(pos + num_ws) != OOR {
            pos += num_ws - 1;
            add_token(pos, &mut prev_end, &mut lens);
            continue;
        }

        // regex: \s+
        if num_ws > 0 {
            pos += num_ws;
            add_token(pos, &mut prev_end, &mut lens);
            continue;
        }

        // no matches
        pos += 1;
        add_token(pos, &mut prev_end, &mut lens);
    }

    // convert codepoint-length words to byte substrings
    let mut words = Vec::with_capacity(lens.len());
    let mut cpt_i = 0usize;
    let mut byte_i = 0usize;
    for &len in &lens {
        let mut nbytes = 0usize;
        for k in 0..len {
            nbytes += cpt_bytes[cpt_i + k];
        }
        words.push(text[byte_i..byte_i + nbytes].to_string());
        cpt_i += len;
        byte_i += nbytes;
    }
    words
}

// ---- glm4 pre-tokenizer split (GLM-4.x / GLM-5.x, llama.cpp LLAMA_VOCAB_PRE_TYPE_CHATGLM4) ----
//
// The zai-org GLM line's `tokenizer.json` declares ONE `Split` regex:
//
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}
//     | ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//
// Verified byte-for-byte off zai-org/GLM-5.3-Flash @ 04c4e9e9 (tokenizer.json sha256
// 19e77364…, the sha banked in that lane's artifact.lock), and the checkpoint's llama.cpp
// `chkhsh` is cdf5f353… — identical to GLM-4.7-Flash's, which upstream registers as `glm4`.
//
// Against `QWEN2_PRETOKENIZE_REGEX` the pattern differs in EXACTLY ONE ATOM: `\p{N}{1,3}`
// instead of `\p{N}`, i.e. digit runs group up to three (the cl100k / LLaMA-3 convention)
// instead of one token per digit.
//
// It is nonetheless a SEPARATE state machine rather than a parameterized `split_qwen35`,
// because `split_qwen35` implements the qwen35 pattern — `[\p{L}\p{M}]+` and
// `[^\s\p{L}\p{M}\p{N}]+` — and llama.cpp deliberately runs qwen2 through those same
// mark-folding classes. The literal GLM/qwen2 classes do NOT fold marks, and the two
// disagree on any text carrying combining marks. Measured against the checkpoint's own
// tokenizer through HF `tokenizers`:
//
//   "cafe\u{301}"     glm4 ["cafe", "\u{301}"]      split_qwen35 ["café"]
//   "x\u{301}y"       glm4 ["x", "\u{301}y"]        split_qwen35 ["x\u{301}y"]
//   "مُحَمَّد"           glm4 ["م","ُح","َم","َّ","د"]     split_qwen35 ["مُحَمَّد"]
//
// So a parameterized reuse would have had to change three class predicates as well as the
// digit arm, and any slip would have moved qwen35/qwen2 ids. A separate machine leaves both
// byte-untouched by construction.
//
// Three class facts, each measured against the real engine (HF `tokenizers`, oniguruma) over
// the full codepoint space rather than assumed:
//   - `\s` is Unicode White_Space and equals memra's WHITESPACE set EXACTLY (25 cpts, no
//     difference in either direction). `a\u{a0}\u{a0}b` -> ["a","\u{a0}","\u{a0}b"] is the
//     decisive case: it only comes out that way if U+00A0 is `\s`.
//   - `\p{N}` includes Nl and No, not just Nd — `①②③④` -> ["①②③","④"]. memra's NUMBER flag
//     already covers them.
//   - `(?i:)` is Unicode simple case folding, so `'ſ` (U+017F) matches the `'s` alternative:
//     `'ſx` -> ["'ſ", "x"]. U+017F is the ONLY codepoint in the whole space that is
//     case-equal to any of s/t/r/e/v/m/l/d beyond their ASCII uppercase forms, and llama's
//     `tolower` map leaves it alone (it is already lowercase), so it is folded explicitly.

/// The one codepoint where oniguruma's `(?i:)` folding and llama's `tolower` map disagree for
/// the contraction alternatives. Measured, not guessed — see the module note above.
const LATIN_SMALL_LETTER_LONG_S: u32 = 0x017F;

/// `tolower`, plus the Unicode simple case folding that llama's lowercase map does not carry.
#[inline]
fn contraction_fold(cpt: u32) -> u32 {
    if cpt == LATIN_SMALL_LETTER_LONG_S {
        's' as u32
    } else {
        tolower(cpt)
    }
}

/// Pre-tokenizer split for the GLM-4.x / GLM-5.x `Split` regex (see the module note above).
///
/// A deterministic codepoint-class scan of the ordered alternation, NOT a regex engine:
/// alternatives are tried leftmost-first at each position and every one of them is
/// anchored-and-bounded, so there is no backtracking and `pos` advances every iteration.
pub fn split_glm4(text: &str) -> Vec<String> {
    let cpts: Vec<u32> = text.chars().map(|c| c as u32).collect();
    let cpt_bytes: Vec<usize> = text.chars().map(|c| c.len_utf8()).collect();
    let n = cpts.len();

    let get_cpt = |pos: usize| -> Option<u32> { cpts.get(pos).copied() };
    let flags_at = |pos: usize| -> CptFlags {
        match cpts.get(pos) {
            Some(&c) => cpt_flags_from_cpt(c),
            None => CptFlags::default(),
        }
    };
    let is_letter = |pos: usize| -> bool { pos < n && flags_at(pos).is_letter() };
    let is_number = |pos: usize| -> bool { pos < n && flags_at(pos).is_number() };
    let is_ws = |pos: usize| -> bool { pos < n && flags_at(pos).is_whitespace() };
    // one member of `[^\s\p{L}\p{N}]` — the complement includes marks, punctuation, symbols,
    // control and UNASSIGNED codepoints; only whitespace/letters/numbers are excluded.
    let is_other = |pos: usize| -> bool {
        pos < n && {
            let f = flags_at(pos);
            !(f.is_whitespace() || f.is_letter() || f.is_number())
        }
    };
    let is_rn = |pos: usize| -> bool {
        matches!(get_cpt(pos), Some(c) if c == '\r' as u32 || c == '\n' as u32)
    };

    let mut lens: Vec<usize> = Vec::new();
    let mut prev_end = 0usize;
    let add_token = |end: usize, prev_end: &mut usize, lens: &mut Vec<usize>| {
        debug_assert!(*prev_end <= end && end <= n);
        if end > *prev_end {
            lens.push(end - *prev_end);
        }
        *prev_end = end;
    };

    let mut pos = 0usize;
    while pos < n {
        // alt 1: (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if get_cpt(pos) == Some('\'' as u32) && pos + 1 < n {
            let c1 = contraction_fold(cpts[pos + 1]);
            if c1 == 's' as u32 || c1 == 't' as u32 || c1 == 'm' as u32 || c1 == 'd' as u32 {
                pos += 2;
                add_token(pos, &mut prev_end, &mut lens);
                continue;
            }
            if pos + 2 < n {
                let c2 = contraction_fold(cpts[pos + 2]);
                if (c1 == 'r' as u32 && c2 == 'e' as u32)
                    || (c1 == 'v' as u32 && c2 == 'e' as u32)
                    || (c1 == 'l' as u32 && c2 == 'l' as u32)
                {
                    pos += 3;
                    add_token(pos, &mut prev_end, &mut lens);
                    continue;
                }
            }
        }

        // alt 2: [^\r\n\p{L}\p{N}]?\p{L}+
        //
        // The optional lead is ONE codepoint and cannot be a letter or a number, so when the
        // current codepoint is a letter the lead is necessarily empty and the run starts here;
        // otherwise the lead may absorb this codepoint (unless it is \r, \n or a number) and
        // the run must start at the next one. `\p{L}+` is letters ONLY — a combining mark ends
        // the run and falls through to alt 4, which is where this family parts company with
        // the qwen35 machine.
        let letters_from = if is_letter(pos) {
            Some(pos)
        } else if !(is_rn(pos) || is_number(pos)) && is_letter(pos + 1) {
            Some(pos + 1)
        } else {
            None
        };
        if let Some(start) = letters_from {
            let mut end = start;
            while is_letter(end) {
                end += 1;
            }
            pos = end;
            add_token(pos, &mut prev_end, &mut lens);
            continue;
        }

        // alt 3: \p{N}{1,3} — the ONE atom that differs from qwen2's `\p{N}`.
        if is_number(pos) {
            let mut end = pos + 1;
            while end < pos + 3 && is_number(end) {
                end += 1;
            }
            pos = end;
            add_token(pos, &mut prev_end, &mut lens);
            continue;
        }

        // alt 4: ` ?[^\s\p{L}\p{N}]+[\r\n]*`
        //
        // The optional lead is a LITERAL space, and the body must be non-empty; a space with
        // no complement codepoint after it therefore fails this alternative outright (the
        // empty-lead retry cannot match either, since a space is `\s`) and falls to the
        // whitespace alternatives below.
        let body_from = if get_cpt(pos) == Some(' ' as u32) && is_other(pos + 1) {
            Some(pos + 1)
        } else if is_other(pos) {
            Some(pos)
        } else {
            None
        };
        if let Some(start) = body_from {
            let mut end = start;
            while is_other(end) {
                end += 1;
            }
            while is_rn(end) {
                end += 1;
            }
            pos = end;
            add_token(pos, &mut prev_end, &mut lens);
            continue;
        }

        // The three whitespace alternatives share one scan of the run.
        let mut num_ws = 0usize;
        let mut last_end_rn = 0usize;
        while is_ws(pos + num_ws) {
            if is_rn(pos + num_ws) {
                last_end_rn = pos + num_ws + 1;
            }
            num_ws += 1;
        }

        // alt 5: \s*[\r\n]+ — greedy `\s*` backtracks to the LAST \r/\n in the run.
        if last_end_rn > 0 {
            pos = last_end_rn;
            add_token(pos, &mut prev_end, &mut lens);
            continue;
        }
        // alt 6: \s+(?!\S) — the run is followed by a non-whitespace codepoint (the scan only
        // stops on one, or on end-of-text), so the lookahead forces giving one back.
        if num_ws > 1 && pos + num_ws < n {
            pos += num_ws - 1;
            add_token(pos, &mut prev_end, &mut lens);
            continue;
        }
        // alt 7: \s+
        if num_ws > 0 {
            pos += num_ws;
            add_token(pos, &mut prev_end, &mut lens);
            continue;
        }

        // No alternative matched. Unreachable — every codepoint is whitespace, a letter, a
        // number, or a member of alt 4's complement — but a pre-tokenizer must never drop
        // bytes, so emit the codepoint as its own word rather than stalling.
        pos += 1;
        add_token(pos, &mut prev_end, &mut lens);
    }

    let mut words = Vec::with_capacity(lens.len());
    let mut cpt_i = 0usize;
    let mut byte_i = 0usize;
    for &len in &lens {
        let mut nbytes = 0usize;
        for k in 0..len {
            nbytes += cpt_bytes[cpt_i + k];
        }
        words.push(text[byte_i..byte_i + nbytes].to_string());
        cpt_i += len;
        byte_i += nbytes;
    }
    words
}

// ---- deepseek-v3 pre-tokenizer split --------------------------------------------------
//
// Step-3.7-Flash's `tokenizer.ggml.pre` is `deepseek-v3`
// (llama.cpp `LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM`). Upstream has NO custom state machine for
// it — it runs three regexes through the generic `std::regex` path, in order, each pass
// subdividing the previous pass's offsets. This is a hand-written equivalent.
//
// The three patterns (`src/llama-vocab.cpp:318-325`):
//   1. "\p{N}{1,3}"
//   2. "[一-龥぀-ゟ゠-ヿ]+"                       (CJK ideographs + hiragana + katakana)
//   3. "[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+
//       |[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
//       | ?[\p{P}\p{S}]+[\r\n]*
//       |\s*[\r\n]+ |\s+(?!\S) |\s+"
//
// Differences from qwen2/qwen35 that make a fall-through silently wrong: digits group in runs
// of up to 3 (not one per token), CJK/kana is its own isolated pass, the letter-run's optional
// leading character excludes \p{P}/\p{S} instead of \p{N}, the non-letter run is
// \p{P}/\p{S}-only (so undefined/control codepoints are NOT absorbed into it), and there is no
// contraction ('s/'t/...) alternative at all.
//
// Two upstream details that are easy to get wrong and are load-bearing here:
//   - Pass 3 runs on the COLLAPSED text (one byte per codepoint, non-ASCII mapped to a
//     category byte), so `\s` is std::regex's ASCII-only \s and every non-ASCII whitespace
//     codepoint has already become 0x0B (which IS ASCII \s). A codepoint carrying two category
//     bits collapses to the 0xD0 fallback and therefore matches NONE of \p{L}/\p{P}/\p{S}.
//   - `unicode_regex_split_stl` emits the GAP before each match as its own word, so a pass
//     never drops text; unmatched spans survive to the next pass.

/// Collapsed-text category byte for one codepoint (`unicode_regex_split`'s `k_ucat_cpt`).
#[inline]
fn collapse_cpt(cpt: u32) -> u8 {
    if cpt < 128 {
        return cpt as u8;
    }
    let fl = cpt_flags_from_cpt(cpt);
    if fl.is_whitespace() {
        return 0x0B; // <vertical tab> — llama's non-ASCII whitespace stand-in
    }
    match fl.category_flag() {
        FLAG_NUMBER => 0xD1,
        FLAG_LETTER => 0xD2,
        flag::PUNCTUATION => 0xD3,
        FLAG_ACCENT_MARK => 0xD4,
        flag::SYMBOL => 0xD5,
        _ => 0xD0, // undefined/separator/control, or any multi-category codepoint
    }
}

/// Classes over the collapsed byte alphabet. Each is `k_ucat_cpt[cat]` plus `k_ucat_map[cat]`
/// (the sub-128 codepoints of that category), exactly as the collapsed regex is built.
#[inline]
fn c_is_letter(b: u8) -> bool {
    b == 0xD2 || b.is_ascii_alphabetic()
}
#[inline]
fn c_is_mark(b: u8) -> bool {
    b == 0xD4 // no sub-128 accent marks
}
#[inline]
fn c_is_punct(b: u8) -> bool {
    b == 0xD3
        || matches!(b,
            0x21..=0x23 | 0x25..=0x2A | 0x2C..=0x2F | 0x3A..=0x3B | 0x3F..=0x40
            | 0x5B..=0x5D | 0x5F | 0x7B | 0x7D)
}
/// DELIBERATE divergence from upstream llama.cpp: `0x7E` (`~`, U+007E, category Sm) is
/// included here but MISSING from upstream's `k_ucat_map` SYMBOL expansion
/// (``"$+<=>^`|"``, `unicode.cpp:1244`, verified on master 2026-08-07) — the single
/// printable-ASCII codepoint where that map disagrees with real Unicode P/S (enumerated
/// over 0x21..0x7E). The HF reference tokenizer's `\p{S}` DOES match `~`, so upstream
/// splits `" ~"` as `[" ", "~"]` while the tokenizer the model was TRAINED with produces
/// `[" ~"]` (one pre-token, `Ġ~`). memra matches the training-time ground truth; receipt:
/// `research/step-sku-20260807/raw/tok-parity-20260807T0640Z.log` (the one corpus
/// mismatch before this fix, `symbols-spaced`, id 6883 `Ġ~` vs `223,96`).
#[inline]
fn c_is_symbol(b: u8) -> bool {
    b == 0xD5 || matches!(b, 0x24 | 0x2B | 0x3C..=0x3E | 0x5E | 0x60 | 0x7C | 0x7E)
}
#[inline]
fn c_is_number(b: u8) -> bool {
    b == 0xD1 || b.is_ascii_digit()
}
/// std::regex `\s` over the collapsed alphabet: ASCII space/\t/\n/\v/\f/\r only (every
/// non-ASCII whitespace codepoint is already 0x0B).
#[inline]
fn c_is_space(b: u8) -> bool {
    matches!(b, 0x20 | 0x09..=0x0D)
}
/// The ASCII punctuation literal class of pattern 3's first alternative,
/// `[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]` — written out because it is NOT the same set as
/// `\p{P}` ∪ `\p{S}` collapsed (it is ASCII-literal and excludes the 0xD3/0xD5 category bytes).
#[inline]
fn c_is_ascii_punct_lit(b: u8) -> bool {
    matches!(b,
        0x21..=0x2F | 0x3A..=0x40 | 0x5B..=0x60 | 0x7B..=0x7E)
}

/// Emit a word boundary of `len` codepoints (the `_add_token` of the custom splitters).
#[inline]
fn push_len(lens: &mut Vec<usize>, len: usize) {
    if len > 0 {
        lens.push(len);
    }
}

/// One `unicode_regex_split_stl` pass: apply `matcher` inside each existing offset window,
/// emitting the gap before each match and then the match itself. `matcher(win_start, pos)`
/// returns the end index of a match starting at `pos`, or `None`.
fn split_pass<F>(offsets: &[usize], mut matcher: F) -> Vec<usize>
where
    F: FnMut(usize, usize, usize) -> Option<usize>,
{
    let mut out: Vec<usize> = Vec::with_capacity(offsets.len());
    let mut start = 0usize;
    for &off in offsets {
        let end = start + off;
        let mut gap = start; // start of the not-yet-emitted gap
        let mut pos = start;
        while pos < end {
            match matcher(start, end, pos) {
                Some(m_end) if m_end > pos => {
                    push_len(&mut out, pos - gap);
                    push_len(&mut out, m_end - pos);
                    gap = m_end;
                    pos = m_end;
                }
                // zero-width or no match: advance the scan, the span stays in the gap.
                _ => pos += 1,
            }
        }
        push_len(&mut out, end - gap);
        start = end;
    }
    out
}

/// Pattern 2's class: `[一-龥぀-ゟ゠-ヿ]` — CJK unified ideographs (U+4E00..U+9FA5, note the
/// upper bound is 龥 not 鿿), hiragana (U+3040..U+309F), katakana (U+30A0..U+30FF).
#[inline]
fn is_cjk_kana(cpt: u32) -> bool {
    (0x4E00..=0x9FA5).contains(&cpt)
        || (0x3040..=0x309F).contains(&cpt)
        || (0x30A0..=0x30FF).contains(&cpt)
}

/// Port of the `deepseek-v3` (DEEPSEEK3_LLM) pre-tokenizer split. Returns the pre-token word
/// substrings of `text`, in order, concatenating back to `text` exactly.
///
/// Cross-checked against an independent reference implementation driven by a different regex
/// engine: `research/step37-p2-20260806/pretok-ref-deepseek-v3.py`.
pub fn split_deepseek_v3(text: &str) -> Vec<String> {
    let cpts: Vec<u32> = text.chars().map(|c| c as u32).collect();
    let cpt_bytes: Vec<usize> = text.chars().map(|c| c.len_utf8()).collect();
    let n = cpts.len();
    let coll: Vec<u8> = cpts.iter().map(|&c| collapse_cpt(c)).collect();

    // ---- pass 1: \p{N}{1,3} (greedy, up to 3) ----
    let mut offsets = vec![n];
    offsets = split_pass(&offsets, |_s, end, pos| {
        if !c_is_number(coll[pos]) {
            return None;
        }
        let mut e = pos + 1;
        while e < end && e - pos < 3 && c_is_number(coll[e]) {
            e += 1;
        }
        Some(e)
    });

    // ---- pass 2: [CJK|kana]+ (runs on the codepoint text: no \p{} class, non-ASCII literals) ----
    offsets = split_pass(&offsets, |_s, end, pos| {
        if !is_cjk_kana(cpts[pos]) {
            return None;
        }
        let mut e = pos + 1;
        while e < end && is_cjk_kana(cpts[e]) {
            e += 1;
        }
        Some(e)
    });

    // ---- pass 3: the six-alternative pattern, leftmost-first (std::regex ECMAScript order) ----
    offsets = split_pass(&offsets, |_s, end, pos| {
        let b = coll[pos];

        // alt 1: [ASCII punct literal][A-Za-z]+
        if c_is_ascii_punct_lit(b) && pos + 1 < end && coll[pos + 1].is_ascii_alphabetic() {
            let mut e = pos + 2;
            while e < end && coll[e].is_ascii_alphabetic() {
                e += 1;
            }
            return Some(e);
        }

        // alt 2: [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
        // the optional lead is any single cpt that is NOT \r \n letter punct symbol; then one or
        // more letter/mark. Try WITH the lead first (leftmost-longest is not the rule here, but
        // ECMAScript takes the first alternative that matches at all and `X?Y+` is greedy on X).
        {
            let lead_ok =
                b != b'\r' && b != b'\n' && !c_is_letter(b) && !c_is_punct(b) && !c_is_symbol(b);
            let mut e = pos;
            if lead_ok && pos + 1 < end && (c_is_letter(coll[pos + 1]) || c_is_mark(coll[pos + 1]))
            {
                e = pos + 1;
            } else if !(c_is_letter(b) || c_is_mark(b)) {
                e = usize::MAX; // no viable letter run here
            }
            if e != usize::MAX {
                let run_start = e;
                while e < end && (c_is_letter(coll[e]) || c_is_mark(coll[e])) {
                    e += 1;
                }
                if e > run_start {
                    return Some(e);
                }
            }
        }

        // alt 3: ' ?[\p{P}\p{S}]+[\r\n]*'
        {
            let mut e = pos;
            if b == b' ' {
                e += 1;
            }
            let run_start = e;
            while e < end && (c_is_punct(coll[e]) || c_is_symbol(coll[e])) {
                e += 1;
            }
            if e > run_start {
                while e < end && (coll[e] == b'\r' || coll[e] == b'\n') {
                    e += 1;
                }
                return Some(e);
            }
        }

        // alt 4: \s*[\r\n]+
        if c_is_space(b) {
            // greedy \s* then require at least one \r\n; ECMAScript backtracks, so find the
            // LAST \r/\n reachable through an unbroken whitespace run.
            let mut e = pos;
            let mut last_rn = None;
            while e < end && c_is_space(coll[e]) {
                if coll[e] == b'\r' || coll[e] == b'\n' {
                    last_rn = Some(e + 1);
                }
                e += 1;
            }
            if let Some(rn_end) = last_rn {
                return Some(rn_end);
            }
            // alt 5: \s+(?!\S) — a whitespace run that is not followed by a non-space char.
            // With no \r\n in the run: if the run is followed by a non-space, backtrack one
            // codepoint so the lookahead sees whitespace; else take the whole run.
            let run_end = e;
            if run_end < end {
                if run_end - pos > 1 {
                    return Some(run_end - 1);
                }
                // single space followed by a non-space: alts 4/5 fail, alt 6 \s+ takes it.
                return Some(pos + 1);
            }
            return Some(run_end); // alt 5 at end-of-window
        }

        None
    });

    // ---- codepoint-length words back to byte substrings ----
    let mut words = Vec::with_capacity(offsets.len());
    let mut cpt_i = 0usize;
    let mut byte_i = 0usize;
    for &len in &offsets {
        let nbytes: usize = cpt_bytes[cpt_i..cpt_i + len].iter().sum();
        words.push(text[byte_i..byte_i + nbytes].to_string());
        cpt_i += len;
        byte_i += nbytes;
    }
    words
}

#[cfg(test)]
mod tests {
    use super::*;

    // generated by research/step37-p2-20260806/pretok-ref-deepseek-v3.py --rust
    const DS3_CASES: &[(&str, &[&str])] = &[
        ("Hello world", &["Hello", " world"]),
        ("Hello, world!", &["Hello", ",", " world", "!"]),
        (
            " leading and trailing ",
            &[" leading", " and", " trailing", " "],
        ),
        (
            "don't can't we're I've I'm you'll he'd",
            &[
                "don", "'t", " can", "'t", " we", "'re", " I", "'ve", " I", "'m", " you", "'ll",
                " he", "'d",
            ],
        ),
        ("1234567 89 0", &["123", "456", "7", " ", "89", " ", "0"]),
        (
            "v0.71.0 and 128K ctx",
            &[
                "v", "0", ".", "71", ".", "0", " and", " ", "128", "K", " ctx",
            ],
        ),
        (
            "Step-3.7-Flash: 196B-A11B (45 blocks)",
            &[
                "Step", "-", "3", ".", "7", "-Flash", ":", " ", "196", "B", "-A", "11", "B", " (",
                "45", " blocks", ")",
            ],
        ),
        (
            "line1\nline2\r\nline3",
            &["line", "1", "\n", "line", "2", "\r\n", "line", "3"],
        ),
        (
            "trailing newlines\n\n\n",
            &["trailing", " newlines", "\n\n\n"],
        ),
        (
            "tabs\tand\t\tspaces   x",
            &["tabs", "\tand", "\t", "\tspaces", "  ", " x"],
        ),
        ("\n\n  \n indented", &["\n\n  \n", " indented"]),
        ("中文测试", &["中文测试"]),
        (
            "混合 English 中文 123",
            &["混合", " English", " ", "中文", " ", "123"],
        ),
        (
            "日本語のテスト、カタカナ",
            &["日本語のテスト", "、", "カタカナ"],
        ),
        ("한국어 테스트", &["한국어", " 테스트"]),
        (
            "emoji 🚀 and symbols ~ ^ | $ +",
            &[
                "emoji", " 🚀", " and", " symbols", " ~", " ^", " |", " $", " +",
            ],
        ),
        ("naïve café résumé", &["naïve", " café", " résumé"]),
        ("Ünïcödé mÄrks", &["Ünïcödé", " mÄrks"]),
        ("áb̧c", &["áb̧c"]),
        (
            "MoE top-8 288 experts@4096",
            &[
                "MoE", " top", "-", "8", " ", "288", " experts", "@", "409", "6",
            ],
        ),
        ("  ", &["  "]),
        (" ", &[" "]),
        ("", &[]),
        ("\t", &["\t"]),
        ("\n", &["\n"]),
        ("x", &["x"]),
        ("@#$%^&*()", &["@#$%^&*()"]),
        (
            "snake_case camelCase kebab-case",
            &["snake", "_case", " camelCase", " kebab", "-case"],
        ),
        ("path/to/file.gguf", &["path", "/to", "/file", ".gguf"]),
        (
            "{\"key\": [1, 2, 3]}",
            &[
                "{\"", "key", "\":", " [", "1", ",", " ", "2", ",", " ", "3", "]}",
            ],
        ),
        (
            "5e6 vs 1e4 rope base",
            &["5", "e", "6", " vs", " ", "1", "e", "4", " rope", " base"],
        ),
        ("ЖИВЁТ русский текст", &["ЖИВЁТ", " русский", " текст"]),
        ("Ελληνικά κείμενα", &["Ελληνικά", " κείμενα"]),
        ("العربية نص", &["العربية", " نص"]),
        ("▁escaped▁space", &["▁", "escaped", "▁", "space"]),
        (
            "100%% sure? yes!!!",
            &["100", "%%", " sure", "?", " yes", "!!!"],
        ),
        (" .a", &[" .", "a"]),
        (" a", &[" a"]),
        ("..", &[".."]),
        ("a1", &["a", "1"]),
        ("1a", &["1", "a"]),
        ("12345678901234", &["123", "456", "789", "012", "34"]),
        (" 123", &[" ", "123"]),
        ("123 ", &["123", " "]),
        ("  123  ", &["  ", "123", "  "]),
        ("-abc", &["-abc"]),
        ("-abc1", &["-abc", "1"]),
        ("~abc", &["~abc"]),
        ("~", &["~"]),
        ("~ ^", &["~", " ^"]),
        (" nbsp", &[" nbsp"]),
        ("a  b", &["a", " ", " b"]),
        ("x \n y", &["x", " \n", " y"]),
        ("x  \n\n  y", &["x", "  \n\n", " ", " y"]),
        ("end with space ", &["end", " with", " space", " "]),
        ("end with spaces   ", &["end", " with", " spaces", "   "]),
        ("\r", &["\r"]),
        ("\r\r\n\n", &["\r\r\n\n"]),
        (" \n ", &[" \n", " "]),
        ("́leading mark", &["́leading", " mark"]),
        ("中1文2", &["中", "1", "文", "2"]),
        ("ーヽヾ", &["ーヽヾ"]),
        ("龥龦", &["龥", "龦"]),
        ("぀〿", &["぀", "〿"]),
    ];

    /// `split_deepseek_v3` vs an independent reference driven by a DIFFERENT regex engine
    /// (`research/step37-p2-20260806/pretok-ref-deepseek-v3.py`, which executes llama.cpp's
    /// three-pass collapsed-text algorithm through Python's `re`). Corpus covers the cases where
    /// deepseek-v3 diverges from qwen2/qwen35: digit grouping in runs of <=3, the isolated
    /// CJK/kana pass, accented letter runs, the punct/symbol-only run, whitespace/newline runs,
    /// and the absence of any contraction alternative.
    #[test]
    fn deepseek_v3_split_matches_reference() {
        for (text, want) in DS3_CASES {
            let got = split_deepseek_v3(text);
            assert_eq!(got, *want, "split_deepseek_v3({text:?})");
            // a pre-tokenizer must never drop or reorder bytes
            assert_eq!(got.concat(), *text, "reassembly of {text:?}");
        }
    }

    /// Why the fall-through was a real bug, mechanism by mechanism. Each case below is one
    /// concrete way `split_qwen35` mis-splits deepseek-v3 text — if any of these ever start
    /// agreeing, the corresponding alternative has been ported wrong (or dropped).
    #[test]
    fn deepseek_v3_differs_from_qwen35_per_mechanism() {
        let q = |t: &str| split_qwen35(t);
        let d = |t: &str| split_deepseek_v3(t);

        // \p{N}{1,3} vs \p{N}: digits group in runs of up to three, left to right.
        assert_eq!(d("12345678901234"), ["123", "456", "789", "012", "34"]);
        assert_eq!(q("1234").len(), 4, "qwen35 emits one token per digit");

        // The isolated CJK/kana pass splits a kana/CJK run away from following punctuation
        // and from a preceding space that qwen35 would absorb.
        assert_eq!(
            d("日本語のテスト、カタカナ"),
            ["日本語のテスト", "、", "カタカナ"]
        );
        assert_eq!(
            q("日本語のテスト、カタカナ"),
            ["日本語のテスト", "、カタカナ"]
        );
        assert_eq!(
            d(" 中文"),
            [" ", "中文"],
            "pass 2 runs before the letter alternative"
        );
        assert_eq!(q(" 中文"), [" 中文"]);

        // alt 3 is [\p{P}\p{S}]+ only, so a codepoint in NEITHER class (U+2581 ▁, category So?
        // no — it is \p{S}o... U+2581 is LOWER ONE EIGHTH BLOCK, \p{So}) still differs because
        // qwen35's alternative is the complement [^\s\p{L}\p{M}\p{N}] and absorbs the
        // following letters' leading position differently.
        assert_eq!(d("▁escaped▁space"), ["▁", "escaped", "▁", "space"]);
        assert_eq!(q("▁escaped▁space"), ["▁escaped", "▁space"]);

        // alt 3 is ' ?[\p{P}\p{S}]+' — a STRICT class, unlike qwen35's complement
        // [^\s\p{L}\p{N}]+ which also swallows format/control codepoints. ZWSP (U+200B, Cf)
        // is in neither \p{P} nor \p{S}, so deepseek-v3 leaves it as its own gap word while
        // qwen35 absorbs it into the punct run.
        assert_eq!(d("a\u{200b}!"), ["a", "\u{200b}", "!"]);
        assert_eq!(q("a\u{200b}!"), ["a", "\u{200b}!"]);
        // '~' IS \p{S} (U+007E, Sm) — upstream llama.cpp's k_ucat_map omits it from the
        // sub-128 SYMBOL expansion; memra deliberately includes it to match the HF
        // training-time tokenizer (see c_is_symbol). Pre-fix this split as
        // [" symbols", " ", "~", " ^"], which is what upstream still produces.
        assert_eq!(d(" symbols ~ ^"), [" symbols", " ~", " ^"]);

        // alt 1 has no counterpart in qwen35 at all: ASCII punctuation immediately followed by
        // ASCII letters is ONE token, and it stops at a non-letter.
        assert_eq!(d("-abc1"), ["-abc", "1"]);

        // No contraction alternative in deepseek-v3 — but the outcome coincides with qwen35
        // here because alt 2's optional lead picks up the apostrophe. Pinned so a future
        // "add the contractions back" edit has to justify itself.
        assert_eq!(d("don't"), ["don", "'t"]);
    }

    /// Banked expectations for `split_glm4`, generated by
    /// `research/glm53-flash-bringup-20260827/pretok-ref-glm4.py --rust`.
    ///
    /// Oracle: the CHECKPOINT'S OWN pre-tokenizer, run through HF `tokenizers` — the engine
    /// `transformers` delegates to and the one that defined the vocab at training time — over
    /// zai-org/GLM-5.3-Flash @ 04c4e9e9's `tokenizer.json`, sha-pinned to
    /// 19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d (the sha in that lane's
    /// `artifact.lock`; the generator refuses to run against any other bytes).
    ///
    /// Coverage is written around the ONE atom that differs from qwen2 — digit runs of every
    /// length 1..12 plus 14, digits against letters/punct/space/newline, and \p{N}'s Nl/No
    /// members (Roman numerals, circled digits, fractions, superscripts) — and around the
    /// alternatives where the LITERAL classes part company with memra's mark-folding qwen35
    /// machine: NFD/NFC accents, leading/interior marks, Arabic harakat, Hebrew niqqud,
    /// Devanagari matras. Then the shared machinery: contractions in both cases (and U+017F,
    /// which oniguruma's `(?i:)` folds onto 's'), whitespace and newline runs incl. NBSP /
    /// ideographic space / U+2028 and the non-`\s` lookalikes U+180E and ZWSP, CJK, emoji
    /// (ZWJ, skin tone, regional indicators), mixed scripts, and code/markup shapes. Then the
    /// branch the pre-tokenizer split does NOT own but every real request takes: GLM's own
    /// control-token literals (`[gMASK]<sop>`, the role turns, think/tool/arg/code-FIM pairs)
    /// and three renders of the checkpoint's own `chat_template.jinja`.
    ///
    /// The wider run this table excerpts covered 526 cases — these plus a 400-case
    /// deterministic fuzz layer and three 3000-char uniform runs — with ZERO split mismatches
    /// and 526/526 token-id parity in both `add_special` modes; `split_qwen35` on the same
    /// corpus mismatches 337/526. Receipts: the lane's `parity-evidence/`.
    #[rustfmt::skip]
    const GLM4_CASES: &[(&str, &[&str])] = &[
        // empty
        ("", &[]),
        // single-letter
        ("x", &["x"]),
        // ascii-hello
        ("Hello, world!", &["Hello", ",", " world", "!"]),
        // leading-space-word
        (" hello", &[" hello"]),
        // sentence
        ("The quick brown fox jumps over the lazy dog.", &["The", " quick", " brown", " fox", " jumps", " over", " the", " lazy", " dog", "."]),
        // digits-len1
        ("1", &["1"]),
        // digits-len2
        ("12", &["12"]),
        // digits-len3
        ("123", &["123"]),
        // digits-len4
        ("1234", &["123", "4"]),
        // digits-len5
        ("12345", &["123", "45"]),
        // digits-len6
        ("123456", &["123", "456"]),
        // digits-len7
        ("1234567", &["123", "456", "7"]),
        // digits-len8
        ("12345678", &["123", "456", "78"]),
        // digits-len9
        ("123456789", &["123", "456", "789"]),
        // digits-len10
        ("1234567890", &["123", "456", "789", "0"]),
        // digits-len11
        ("12345678901", &["123", "456", "789", "01"]),
        // digits-len12
        ("123456789012", &["123", "456", "789", "012"]),
        // digits-14
        ("12345678901234", &["123", "456", "789", "012", "34"]),
        // digits-in-words
        ("abc123def4567gh89", &["abc", "123", "def", "456", "7", "gh", "89"]),
        // digit-letter-alternating
        ("a1b22c333d4444e", &["a", "1", "b", "22", "c", "333", "d", "444", "4", "e"]),
        // decimal
        ("3.14159265358979", &["3", ".", "141", "592", "653", "589", "79"]),
        // negative
        ("-273.15 degrees", &["-", "273", ".", "15", " degrees"]),
        // thousands
        ("1,234,567.89", &["1", ",", "234", ",", "567", ".", "89"]),
        // version-string
        ("v2.7.18-rc3+build.4521", &["v", "2", ".", "7", ".", "18", "-rc", "3", "+build", ".", "452", "1"]),
        // date-iso
        ("2026-08-07T06:00:00Z", &["202", "6", "-", "08", "-", "07", "T", "06", ":", "00", ":", "00", "Z"]),
        // phone
        ("+1 (555) 010-4477 ext. 42", &["+", "1", " (", "555", ")", " ", "010", "-", "447", "7", " ext", ".", " ", "42"]),
        // hex-literal
        ("0xDEADBEEF 0b1011 1e-9 6.022e23", &["0", "xDEADBEEF", " ", "0", "b", "101", "1", " ", "1", "e", "-", "9", " ", "6", ".", "022", "e", "23"]),
        // digits-then-newline
        ("123\n456", &["123", "\n", "456"]),
        // space-then-digits
        (" 1234", &[" ", "123", "4"]),
        // digits-then-space
        ("1234 ", &["123", "4", " "]),
        // digits-hugging-punct
        ("(1234)[5678]{9012}", &["(", "123", "4", ")[", "567", "8", "]{", "901", "2", "}"]),
        // arabic-indic-digits
        ("١٢٣٤٥٦٧", &["١٢٣", "٤٥٦", "٧"]),
        // fullwidth-digits
        ("１２３４５", &["１２３", "４５"]),
        // math-bold-digits
        ("𝟎𝟏𝟐𝟑", &["𝟎𝟏𝟐", "𝟑"]),
        // roman-numerals-Nl
        ("ⅨⅨⅨⅨ", &["ⅨⅨⅨ", "Ⅸ"]),
        // circled-digits-No
        ("①②③④", &["①②③", "④"]),
        // fractions-No
        ("½½½½", &["½½½", "½"]),
        // superscript-No
        ("x²²²² + y³", &["x", "²²²", "²", " +", " y", "³"]),
        // mixed-script-digits
        ("12٣٤ 5６", &["12٣", "٤", " ", "5６"]),
        // contractions-lower
        ("don't can't we're I've I'm you'll he'd", &["don", "'t", " can", "'t", " we", "'re", " I", "'ve", " I", "'m", " you", "'ll", " he", "'d"]),
        // contractions-upper
        ("DON'T CAN'T WE'RE I'VE I'M YOU'LL HE'D", &["DON", "'T", " CAN", "'T", " WE", "'RE", " I", "'VE", " I", "'M", " YOU", "'LL", " HE", "'D"]),
        // contractions-mixed
        ("We'Ve a'lL It'S tHeY'rE", &["We", "'Ve", " a", "'lL", " It", "'S", " tHeY", "'rE"]),
        // contraction-long-s
        ("'ſx and 'ſ", &["'ſ", "x", " and", " '", "ſ"]),
        // apostrophe-not-contraction
        ("'q 'z '9 ' '", &["'q", " '", "z", " '", "9", " '", " '"]),
        // quote-then-word
        ("'quoted' \"double\"", &["'quoted", "'", " \"", "double", "\""]),
        // nfd-e-acute
        ("café", &["cafe", "́"]),
        // nfc-e-acute
        ("café", &["café"]),
        // mark-leading
        ("́abc", &["́abc"]),
        // mark-interior
        ("x́y", &["x", "́y"]),
        // mark-runs
        ("áb́ć", &["a", "́b", "́c", "́"]),
        // mark-double
        ("á̈b", &["a", "́̈", "b"]),
        // arabic-harakat
        ("مُحَمَّد", &["م", "ُح", "َم", "َّ", "د"]),
        // hebrew-niqqud
        ("שָׁלוֹם", &["ש", "ָׁ", "לו", "ֹם"]),
        // devanagari-matras
        ("हिन्दी", &["ह", "िन", "्द", "ी"]),
        // mark-then-digit
        ("á1234", &["a", "́", "123", "4"]),
        // mark-then-space
        ("á b", &["a", "́", " b"]),
        // leading-trailing-spaces
        ("   leading and trailing spaces   ", &["  ", " leading", " and", " trailing", " spaces", "   "]),
        // interior-double-space
        ("a  b", &["a", " ", " b"]),
        // interior-triple-space
        ("a   b", &["a", "  ", " b"]),
        // space-only-1
        (" ", &[" "]),
        // space-only-2
        ("  ", &["  "]),
        // space-only-8
        ("        ", &["        "]),
        // tabs
        ("tabs\tand\t\tspaces   x", &["tabs", "\tand", "\t", "\tspaces", "  ", " x"]),
        // newline-single
        ("\n", &["\n"]),
        // newline-run
        ("\n\n\n", &["\n\n\n"]),
        // crlf
        ("line1\r\nline2\r\n\r\nline4", &["line", "1", "\r\n", "line", "2", "\r\n\r\n", "line", "4"]),
        // cr-only
        ("\r\r\n\n", &["\r\r\n\n"]),
        // ws-then-newline
        ("x  \n\n  y", &["x", "  \n\n", " ", " y"]),
        // newline-then-ws
        ("\n\n  \n indented", &["\n\n  \n", " indented"]),
        // trailing-newlines
        ("trailing newlines\n\n\n", &["trailing", " newlines", "\n\n\n"]),
        // space-before-eof
        ("end with space ", &["end", " with", " space", " "]),
        // nbsp-single
        ("a\u{a0}b", &["a", "\u{a0}b"]),
        // nbsp-double
        ("a\u{a0}\u{a0}b", &["a", "\u{a0}", "\u{a0}b"]),
        // ideographic-space
        ("a\u{3000}\u{3000}b", &["a", "\u{3000}", "\u{3000}b"]),
        // line-separator
        ("a\u{2028}\u{2028}b", &["a", "\u{2028}", "\u{2028}b"]),
        // mongolian-vowel-sep
        ("a\u{180e}\u{180e}b", &["a", "\u{180e}\u{180e}", "b"]),
        // zwsp
        ("a\u{200b}b", &["a", "\u{200b}b"]),
        // form-feed-vtab
        ("a\u{c}\u{b}b", &["a", "\u{c}", "\u{b}b"]),
        // cjk-sentence
        ("中文测试", &["中文测试"]),
        // cjk-mixed-digits
        ("中1文2", &["中", "1", "文", "2"]),
        // japanese
        ("日本語のテスト、カタカナ", &["日本語のテスト", "、カタカナ"]),
        // korean
        ("한국어 테스트", &["한국어", " 테스트"]),
        // cyrillic
        ("ЖИВЁТ русский", &["ЖИВЁТ", " русский"]),
        // greek
        ("Ελληνικά κείμενα", &["Ελληνικά", " κείμενα"]),
        // arabic
        ("العربية نص", &["العربية", " نص"]),
        // mixed-scripts
        ("混合 English 中文 123 рус", &["混合", " English", " 中文", " ", "123", " рус"]),
        // thai
        ("ภาษาไทย 123", &["ภาษาไทย", " ", "123"]),
        // emoji-run
        ("🚀🔥✅", &["🚀🔥✅"]),
        // emoji-zwj
        ("😶\u{200d}🌫️", &["😶\u{200d}🌫️"]),
        // emoji-with-text
        ("emoji test 🚀🔥✅ and math ∑∫√π≠≤", &["emoji", " test", " 🚀🔥✅", " and", " math", " ∑∫√", "π", "≠≤"]),
        // skin-tone
        ("👍🏽", &["👍🏽"]),
        // regional-indicators
        ("🇺🇸🇨🇳", &["🇺🇸🇨🇳"]),
        // symbols-spaced
        ("symbols ~ ^ | $ + = < >", &["symbols", " ~", " ^", " |", " $", " +", " =", " <", " >"]),
        // punct-run
        ("@#$%^&*()", &["@#$%^&*()"]),
        // punct-heavy
        ("''''''```````\"\"\"\"......!!!!!!??????", &["''''''```````\"\"\"\"......!!!!!!??????"]),
        // lower-eighth-block
        ("▁escaped▁space", &["▁escaped", "▁space"]),
        // unassigned-tag-cpt
        ("a\u{e0001}b", &["a", "\u{e0001}b"]),
        // rust-fn
        ("fn main() { let x: i32 = 42; println!(\"{}\", x*2); }", &["fn", " main", "()", " {", " let", " x", ":", " i", "32", " =", " ", "42", ";", " println", "!(\"{}\",", " x", "*", "2", ");", " }"]),
        // json
        ("{\"key\": [1, 2, 3], \"n\": 12345}", &["{\"", "key", "\":", " [", "1", ",", " ", "2", ",", " ", "3", "],", " \"", "n", "\":", " ", "123", "45", "}"]),
        // path
        ("path/to/file-00042.gguf", &["path", "/to", "/file", "-", "000", "42", ".gguf"]),
        // identifiers
        ("snake_case camelCase kebab-case SCREAMING_SNAKE_2", &["snake", "_case", " camelCase", " kebab", "-case", " SCREAMING", "_SNAKE", "_", "2"]),
        // markdown-fence
        ("```python\nprint(1234)\n```\n", &["```", "python", "\n", "print", "(", "123", "4", ")\n", "```\n"]),
        // html
        ("<div class=\"x\" id=\"row-17\">text</div>", &["<div", " class", "=\"", "x", "\"", " id", "=\"", "row", "-", "17", "\">", "text", "</", "div", ">"]),
        // chatml
        ("<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n", &["<|", "im", "_start", "|>", "user", "\n", "What", " is", " ", "2", "+", "2", "?<|", "im", "_end", "|>\n", "<|", "im", "_start", "|>", "assistant", "\n"]),
        // llamacpp-chktxt
        ("\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n🚀 (normal) 😶\u{200d}🌫️ (multiple emojis concatenated) ✅ 🦙🦙 3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～ ------======= нещо на Български ''''''```````\"\"\"\"......!!!!!!?????? I've been 'told he's there, 'RE you sure? 'M not sure I'll make it, 'D you like some tea? We'Ve a'lL", &["\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n", "🚀", " (", "normal", ")", " 😶\u{200d}🌫️", " (", "multiple", " emojis", " concatenated", ")", " ✅", " 🦙🦙", " ", "3", " ", "33", " ", "333", " ", "333", "3", " ", "333", "33", " ", "333", "333", " ", "333", "333", "3", " ", "333", "333", "33", " ", "3", ".", "3", " ", "3", "..", "3", " ", "3", "...", "3", " ក", "ាន", "់ត", "ែព", "ិស", "េសអ", "ាច", "😁", " ?", "我想在apple工作", "131", "415", "1", "天", "～", " ------=======", " нещо", " на", " Български", " ''''''```````\"\"\"\"......!!!!!!??????", " I", "'ve", " been", " '", "told", " he", "'s", " there", ",", " '", "RE", " you", " sure", "?", " '", "M", " not", " sure", " I", "'ll", " make", " it", ",", " '", "D", " you", " like", " some", " tea", "?", " We", "'Ve", " a", "'lL"]),
        // st-endoftext
        ("<|endoftext|>", &["<|", "endoftext", "|>"]),
        // st-gmask-sop
        ("[gMASK]<sop>", &["[gMASK", "]<", "sop", ">"]),
        // st-role-turn
        ("<|system|>You are helpful.<|user|>hi 1234<|assistant|>", &["<|", "system", "|>", "You", " are", " helpful", ".<|", "user", "|>", "hi", " ", "123", "4", "<|", "assistant", "|>"]),
        // st-think
        ("<think>reasoning 42</think>answer", &["<think", ">reasoning", " ", "42", "</", "think", ">answer"]),
        // st-tool-call
        ("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>", &["<tool", "_call", ">get", "_weather", "<arg", "_key", ">city", "</", "arg", "_key", "><", "arg", "_value", ">Paris", "</", "arg", "_value", "></", "tool", "_call", ">"]),
        // st-tool-response
        ("<tool_response>{\"temp\": 21}</tool_response>", &["<tool", "_response", ">{\"", "temp", "\":", " ", "21", "}</", "tool", "_response", ">"]),
        // st-observation
        ("<|observation|>result 007<|assistant|>", &["<|", "observation", "|>", "result", " ", "007", "<|", "assistant", "|>"]),
        // st-code-fim
        ("<|code_prefix|>def f(x):<|code_suffix|>return x<|code_middle|>", &["<|", "code", "_prefix", "|>", "def", " f", "(x", "):<|", "code", "_suffix", "|>", "return", " x", "<|", "code", "_middle", "|>"]),
        // st-nothink
        ("/nothink what is 2+2?", &["/nothink", " what", " is", " ", "2", "+", "2", "?"]),
        // st-box
        ("<|begin_of_box|>1234<|end_of_box|>", &["<|", "begin", "_of", "_box", "|>", "123", "4", "<|", "end", "_of", "_box", "|>"]),
        // st-nonspecial-mask
        ("[MASK][sMASK]<eop>", &["[MASK", "][", "sMASK", "]<", "eop", ">"]),
        // st-glued
        ("a<|user|>1<|assistant|>2", &["a", "<|", "user", "|>", "1", "<|", "assistant", "|>", "2"]),
        // st-partial-literal
        ("<|user and |assistant|> and <|nope|>", &["<|", "user", " and", " |", "assistant", "|>", " and", " <|", "nope", "|>"]),
        // st-adjacent-digits
        ("<|user|>1234<|assistant|>5678", &["<|", "user", "|>", "123", "4", "<|", "assistant", "|>", "567", "8"]),
        // tmpl-simple
        ("[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>What is 12345 divided by 3?<|assistant|><think>", &["[gMASK", "]<", "sop", "><|", "system", "|>", "Reasoning", " Effort", ":", " Max", "<|", "user", "|>", "What", " is", " ", "123", "45", " divided", " by", " ", "3", "?<|", "assistant", "|><", "think", ">"]),
        // tmpl-multiturn
        ("[gMASK]<sop><|system|>Reasoning Effort: Max<|system|>You are terse.<|user|>café résumé, 2026-08-28<|assistant|><think></think>Noted: 1,234 items.<|user|>中文 and 🚀 too?<|assistant|><think>", &["[gMASK", "]<", "sop", "><|", "system", "|>", "Reasoning", " Effort", ":", " Max", "<|", "system", "|>", "You", " are", " terse", ".<|", "user", "|>", "café", " résumé", ",", " ", "202", "6", "-", "08", "-", "28", "<|", "assistant", "|><", "think", "></", "think", ">Noted", ":", " ", "1", ",", "234", " items", ".<|", "user", "|>", "中文", " and", " 🚀", " too", "?<|", "assistant", "|><", "think", ">"]),
        // tmpl-tools
        ("[gMASK]<sop><|system|>Reasoning Effort: Max<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n\n{\"name\": \"get_weather\", \"description\": \"Get weather for a city\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}\n\n\n</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call><|user|>weather in Paris?<|assistant|><think>", &["[gMASK", "]<", "sop", "><|", "system", "|>", "Reasoning", " Effort", ":", " Max", "<|", "system", "|>\n", "#", " Tools", "\n\n", "You", " may", " call", " one", " or", " more", " functions", " to", " assist", " with", " the", " user", " query", ".\n\n", "You", " are", " provided", " with", " function", " signatures", " within", " <", "tools", "></", "tools", ">", " XML", " tags", ":\n", "<tools", ">\n\n", "{\"", "name", "\":", " \"", "get", "_weather", "\",", " \"", "description", "\":", " \"", "Get", " weather", " for", " a", " city", "\",", " \"", "parameters", "\":", " {\"", "type", "\":", " \"", "object", "\",", " \"", "properties", "\":", " {\"", "city", "\":", " {\"", "type", "\":", " \"", "string", "\"}},", " \"", "required", "\":", " [\"", "city", "\"]}}\n\n\n", "</", "tools", ">\n\n", "For", " each", " function", " call", ",", " output", " the", " function", " name", " and", " arguments", " within", " the", " following", " XML", " format", ":\n", "<tool", "_call", ">{", "function", "-name", "}<", "arg", "_key", ">{", "arg", "-key", "-", "1", "}</", "arg", "_key", "><", "arg", "_value", ">{", "arg", "-value", "-", "1", "}</", "arg", "_value", "><", "arg", "_key", ">{", "arg", "-key", "-", "2", "}</", "arg", "_key", "><", "arg", "_value", ">{", "arg", "-value", "-", "2", "}</", "arg", "_value", ">...</", "tool", "_call", "><|", "user", "|>", "weather", " in", " Paris", "?<|", "assistant", "|><", "think", ">"]),
    ];
    /// `split_glm4` must equal the checkpoint's own pre-tokenizer, case for case, byte for
    /// byte — everything downstream (BPE merges, ids, goldens, accept counts) is wrong
    /// otherwise. See the corpus note above for the wider run this table excerpts.
    #[test]
    fn glm4_split_matches_reference() {
        for (text, want) in GLM4_CASES {
            let got = split_glm4(text);
            assert_eq!(got, *want, "split_glm4({text:?})");
            // a pre-tokenizer must never drop or reorder bytes
            assert_eq!(got.concat(), *text, "reassembly of {text:?}");
        }
    }

    /// The digit atom, stated on its own so a regression cannot hide inside the big table:
    /// `\p{N}{1,3}` groups left to right in threes, and qwen2/qwen35 emit one token per
    /// digit. If these two ever agree on a multi-digit input, the new variant is pointless
    /// and something has been wired back to the old machine.
    #[test]
    fn glm4_groups_digits_where_qwen35_does_not() {
        for n in 1..=12usize {
            let text: String = "123456789012".chars().take(n).collect();
            let got = split_glm4(&text);
            let want: Vec<String> = text
                .as_bytes()
                .chunks(3)
                .map(|c| String::from_utf8(c.to_vec()).unwrap())
                .collect();
            assert_eq!(got, want, "glm4 digit run of {n}");
            assert_eq!(got.len(), n.div_ceil(3));
            // qwen2/qwen35: one token per digit, always
            assert_eq!(split_qwen35(&text).len(), n, "qwen35 digit run of {n}");
            if n > 3 {
                assert_ne!(split_glm4(&text), split_qwen35(&text), "n={n}");
            }
        }
        // non-ASCII \p{N} groups the same way (Nd, Nl and No all count)
        assert_eq!(
            split_glm4("\u{661}\u{662}\u{663}\u{664}"),
            ["\u{661}\u{662}\u{663}", "\u{664}"]
        );
        assert_eq!(
            split_glm4("\u{2168}\u{2168}\u{2168}\u{2168}"),
            ["\u{2168}\u{2168}\u{2168}", "\u{2168}"]
        );
        assert_eq!(
            split_glm4("\u{2460}\u{2461}\u{2462}\u{2463}"),
            ["\u{2460}\u{2461}\u{2462}", "\u{2463}"]
        );
        // and the run is bounded by the class, not just by three
        assert_eq!(split_glm4("12a"), ["12", "a"]);
        assert_eq!(split_glm4("1a2"), ["1", "a", "2"]);
    }

    /// The SECOND divergence, which the one-atom regex diff does not show: the shared qwen35
    /// machine folds `\p{M}` into its letter runs (llama.cpp deliberately routes qwen2 through
    /// qwen35's mark-including classes), while the GLM/qwen2 pattern's `\p{L}+` stops at a
    /// combining mark and ` ?[^\s\p{L}\p{N}]+` picks it up instead. That is why this is a
    /// separate state machine and not a `bool` on the old one. Every expectation here is the
    /// checkpoint's own tokenizer.
    #[test]
    fn glm4_does_not_fold_combining_marks_like_qwen35() {
        let pairs: &[(&str, &[&str], &[&str])] = &[
            // text, glm4 (= the checkpoint), qwen35 (mark-folding)
            ("cafe\u{301}", &["cafe", "\u{301}"], &["cafe\u{301}"]),
            ("x\u{301}y", &["x", "\u{301}y"], &["x\u{301}y"]),
            (
                "a\u{301}b\u{301}c\u{301}",
                &["a", "\u{301}b", "\u{301}c", "\u{301}"],
                &["a\u{301}b\u{301}c\u{301}"],
            ),
            // Arabic harakat and Devanagari matras are \p{M} and sit mid-word constantly
            (
                "\u{645}\u{64f}\u{62d}\u{64e}\u{645}\u{64e}\u{651}\u{62f}",
                &[
                    "\u{645}",
                    "\u{64f}\u{62d}",
                    "\u{64e}\u{645}",
                    "\u{64e}\u{651}",
                    "\u{62f}",
                ],
                &["\u{645}\u{64f}\u{62d}\u{64e}\u{645}\u{64e}\u{651}\u{62f}"],
            ),
            // Hebrew niqqud, same mechanism
            (
                "\u{5e9}\u{5c1}\u{5b8}\u{5dc}\u{5d5}\u{5b9}\u{5dd}",
                &[
                    "\u{5e9}",
                    "\u{5c1}\u{5b8}",
                    "\u{5dc}\u{5d5}",
                    "\u{5b9}\u{5dd}",
                ],
                &["\u{5e9}\u{5c1}\u{5b8}\u{5dc}\u{5d5}\u{5b9}\u{5dd}"],
            ),
        ];
        for (text, glm, qwen) in pairs {
            assert_eq!(split_glm4(text), *glm, "glm4({text:?})");
            assert_eq!(split_qwen35(text), *qwen, "qwen35({text:?})");
            assert_ne!(split_glm4(text), split_qwen35(text), "{text:?}");
        }
        // a leading mark is the one shape where they agree: alt 2's optional lead takes it
        assert_eq!(split_glm4("\u{301}abc"), ["\u{301}abc"]);
        assert_eq!(split_qwen35("\u{301}abc"), ["\u{301}abc"]);
    }

    /// `(?i:)` in the checkpoint's engine is Unicode simple case folding, so U+017F LATIN
    /// SMALL LETTER LONG S matches the `'s` alternative. llama's `tolower` map leaves U+017F
    /// alone (it is already lowercase), so the fold is explicit in `contraction_fold` — and
    /// U+017F is the only codepoint in the whole space that needs it for these letters.
    #[test]
    fn glm4_contractions_fold_like_the_checkpoint() {
        assert_eq!(split_glm4("'\u{17f}x"), ["'\u{17f}", "x"]);
        assert_eq!(split_glm4("'sx"), ["'s", "x"]);
        assert_eq!(split_glm4("'Sx"), ["'S", "x"]);
        // 'k is NOT a contraction alternative, so U+212A KELVIN SIGN must not become one
        assert_eq!(split_glm4("'\u{212a}x"), ["'\u{212a}x"]);
        assert_eq!(split_glm4("We'Ve a'lL"), ["We", "'Ve", " a", "'lL"]);
        // the pre-existing machine does NOT fold U+017F — pinned so the difference is visible
        assert_eq!(split_qwen35("'\u{17f}x"), ["'\u{17f}x"]);
    }

    /// Everything OUTSIDE the two divergences must be byte-identical to the machine that is
    /// already parity-proven against llama.cpp, or the copy drifted while being written.
    /// Corpus = the deepseek-v3 table's inputs minus the cases carrying digits or combining
    /// marks (which is exactly where the two are SUPPOSED to differ).
    #[test]
    fn glm4_equals_qwen35_off_the_two_divergences() {
        let mut compared = 0usize;
        for (text, _) in DS3_CASES {
            let has_digit = text
                .chars()
                .any(|c| cpt_flags_from_cpt(c as u32).is_number());
            let has_mark = text
                .chars()
                .any(|c| cpt_flags_from_cpt(c as u32).is_accent_mark());
            let has_long_s = text.contains('\u{17f}');
            if has_digit || has_mark || has_long_s {
                continue;
            }
            assert_eq!(
                split_glm4(text),
                split_qwen35(text),
                "glm4 and qwen35 must agree on {text:?}"
            );
            compared += 1;
        }
        assert!(compared >= 30, "corpus shrank to {compared} cases");
    }

    /// Same failure shape as `deepseek_v3_split_survives_long_uniform_runs`: an iterative scan,
    /// no backtracking, no recursion, no dropped bytes on the long uniform runs that arrive
    /// inside tool results.
    #[test]
    fn glm4_split_survives_long_uniform_runs() {
        let cases: Vec<(&str, String)> = vec![
            ("ascii-letter-131k", "Z".repeat(131_072)),
            ("digit-131k", "7".repeat(131_072)),
            ("space-131k", " ".repeat(131_072)),
            ("newline-131k", "\n".repeat(131_072)),
            ("mark-131k", "\u{301}".repeat(131_072)),
            ("cjk-64k", "\u{4e2d}".repeat(65_536)),
            (
                "mixed-runs",
                format!(
                    "{}{}{}{}",
                    "Z".repeat(65_536),
                    " ".repeat(65_536),
                    "7".repeat(65_536),
                    "\n".repeat(65_536)
                ),
            ),
        ];
        for (name, text) in &cases {
            let t0 = std::time::Instant::now();
            let words = split_glm4(text);
            let dt = t0.elapsed();
            assert_eq!(words.concat(), *text, "{name}: reassembly");
            assert!(
                dt < std::time::Duration::from_secs(10),
                "{name}: split took {dt:?}"
            );
        }
        let d = split_glm4(&"7".repeat(131_072));
        assert_eq!(d.len(), 131_072usize.div_ceil(3), "digits group in threes");
        assert!(d.iter().take(d.len() - 1).all(|w| w == "777"));
        let z = "Z".repeat(131_072);
        assert_eq!(
            split_glm4(&z),
            std::slice::from_ref(&z),
            "one letter-run word"
        );
        let s = " ".repeat(131_072);
        assert_eq!(
            split_glm4(&s),
            std::slice::from_ref(&s),
            "one ws-run word (alt 7)"
        );
        let ws_then = format!("{}x", " ".repeat(131_072));
        assert_eq!(
            split_glm4(&ws_then),
            [" ".repeat(131_071), " x".to_string()],
            "\\s+(?!\\S) backtracks one before a non-space"
        );
    }

    /// Sanity on the collapse map itself (`unicode_regex_split`'s k_ucat_cpt).
    #[test]
    fn collapse_map_category_bytes() {
        assert_eq!(collapse_cpt(b'a' as u32), b'a', "ASCII passes through");
        assert_eq!(collapse_cpt(0x4E2D), 0xD2, "CJK ideograph is a LETTER");
        assert_eq!(collapse_cpt(0x00E9), 0xD2, "e-acute is a LETTER");
        assert_eq!(
            collapse_cpt(0x0301),
            0xD4,
            "combining acute is an ACCENT_MARK"
        );
        assert_eq!(
            collapse_cpt(0x3001),
            0xD3,
            "ideographic comma is PUNCTUATION"
        );
        assert_eq!(
            collapse_cpt(0x00A0),
            0x0B,
            "NBSP collapses to the ws stand-in"
        );
        assert_eq!(collapse_cpt(0x0660), 0xD1, "Arabic-Indic digit is a NUMBER");
    }

    /// Regression guard for the llama.cpp #26965 failure shape: upstream runs the
    /// deepseek-v3 pre-tokenizer through backtracking `std::regex`, which stack-overflows
    /// on long uniform ASCII runs inside tool results ('Z' x 131072). memra's port is an
    /// ITERATIVE three-pass scan (`split_pass` closures — no regex engine, no recursion,
    /// `pos` advances every iteration), so the same inputs must complete fast, drop no
    /// bytes, and produce the shape each alternative promises. If this test ever crashes
    /// or trips the time bound, someone reintroduced recursion/backtracking.
    #[test]
    fn deepseek_v3_split_survives_long_uniform_runs() {
        let cases: Vec<(&str, String)> = vec![
            // the issue's exact reproducer shape
            ("ascii-letter-131k", "Z".repeat(131_072)),
            ("ascii-letter-1m", "Z".repeat(1_048_576)),
            ("space-131k", " ".repeat(131_072)),
            ("digit-131k", "7".repeat(131_072)),
            (
                "mixed-runs",
                format!(
                    "{}{}{}{}",
                    "Z".repeat(65_536),
                    " ".repeat(65_536),
                    "7".repeat(65_536),
                    "\n".repeat(65_536)
                ),
            ),
            // non-ASCII runs: collapsed LETTER (0xD2), the pass-2 CJK/kana class, and a
            // collapsed SYMBOL (0xD5) run through alt 3
            ("accented-letter-64k", "é".repeat(65_536)),
            ("cjk-64k", "中".repeat(65_536)),
            ("kana-64k", "ア".repeat(65_536)),
            ("symbol-64k", "\u{2581}".repeat(65_536)),
        ];
        for (name, text) in &cases {
            let t0 = std::time::Instant::now();
            let words = split_deepseek_v3(text);
            let dt = t0.elapsed();
            // a pre-tokenizer must never drop or reorder bytes
            assert_eq!(words.concat(), *text, "{name}: reassembly");
            // linear scan, not quadratic backtracking: even debug builds finish in well
            // under a second per case; 10s catches a blowup without flaking a loaded box.
            assert!(
                dt < std::time::Duration::from_secs(10),
                "{name}: split took {dt:?}"
            );
        }
        // shape spot-checks on the giant runs (not just survival):
        let z = "Z".repeat(131_072);
        assert_eq!(
            split_deepseek_v3(&z),
            std::slice::from_ref(&z),
            "one letter-run word"
        );
        let d = split_deepseek_v3(&"7".repeat(131_072));
        assert_eq!(d.len(), 131_072usize.div_ceil(3), "digits group in threes");
        assert!(d.iter().take(d.len() - 1).all(|w| w == "777"));
        let s = " ".repeat(131_072);
        assert_eq!(
            split_deepseek_v3(&s),
            std::slice::from_ref(&s),
            "one ws-run word (alt 5)"
        );
        // a ws run FOLLOWED by text backtracks one codepoint (the (?!\S) lookahead)
        let ws_then = format!("{}x", " ".repeat(131_072));
        assert_eq!(
            split_deepseek_v3(&ws_then),
            [" ".repeat(131_071), " x".to_string()],
            "\\s+(?!\\S) backtracks one before a non-space"
        );
    }
}
