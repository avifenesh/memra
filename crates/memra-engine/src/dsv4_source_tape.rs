//! The DSv4 gate prompt tape (memra #657).
//!
//! The perf and identity gates prompt with `Review this inference engine source:\n\n{tape}` and
//! consume at most a few thousand tokens of it. The originally pinned tape was cut from a dirty
//! tree at 9e3c8b550 and survives on no reachable machine. `tools/dsv4-source-tape.py` rebuilds
//! the clean tape at that commit, byte-identical to the pinned one for its first
//! `SHARED_PREFIX_BYTES`. Both digests are accepted, and every gate tokenizes only that shared
//! prefix, so either tape yields the same prompt tokens by construction. A mode that reads past
//! the prefix must call [`SourceTape::full_pinned`], which refuses the rebuild.

use sha2::{Digest, Sha256};

/// The tape the gates were calibrated on (22,019,130 bytes, dirty tree at 9e3c8b550).
pub const PINNED_SHA256: &str = "f6e175a6f2588953568746fec0cd43fcd046405f74b5c71ce071fe7f37238ded";
/// `tools/dsv4-source-tape.py` output (22,013,722 bytes, clean tree at 9e3c8b550).
pub const REBUILD_SHA256: &str = "11e4bd80352f4a24504ffdee519b33bbc527b3b9bcfeb531815e5731b190cb8c";
/// First byte at which the two tapes differ (`bin/dsv4_fp4_reduce_gate.rs` in the dirty tree).
pub const SHARED_PREFIX_BYTES: usize = 3_736_115;
/// Token margin a gate prompt keeps below the prefix cut, so a merge at the cut cannot reach it.
const CUT_MARGIN_TOKENS: usize = 4096;

pub struct SourceTape {
    text: String,
    /// The accepted digest this tape matched.
    pub sha256: &'static str,
}

impl SourceTape {
    /// Reads a tape and refuses any digest other than the pinned tape or its rebuild.
    pub fn read(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("source tape {}: {e}", path.display()))?;
        let digest = format!("{:x}", Sha256::digest(text.as_bytes()));
        let sha256 = [PINNED_SHA256, REBUILD_SHA256]
            .into_iter()
            .find(|d| *d == digest)
            .ok_or_else(|| {
                format!(
                    "source tape sha256 {digest} is neither the pinned tape {PINNED_SHA256} nor \
                     its rebuild {REBUILD_SHA256} (tools/dsv4-source-tape.py, memra #657)"
                )
            })?;
        Ok(Self { text, sha256 })
    }

    /// The bytes both tapes share, cut at a char boundary. Gate prompts read only these.
    pub fn shared(&self) -> &str {
        let mut end = SHARED_PREFIX_BYTES.min(self.text.len());
        while !self.text.is_char_boundary(end) {
            end -= 1;
        }
        &self.text[..end]
    }

    /// Tokenizes `{header}{shared prefix}` and asserts the caller's `consumed` tokens stay
    /// `CUT_MARGIN_TOKENS` below the cut, so they are the same tokens the full pinned tape gives.
    pub fn prompt(
        &self,
        tokenizer: &memra_tokenizer::Tokenizer,
        header: &str,
        consumed: usize,
    ) -> Vec<u32> {
        let tokens = tokenizer.encode(&format!("{header}{}", self.shared()), true);
        assert!(
            tokens.len() >= consumed + CUT_MARGIN_TOKENS,
            "gate consumes {consumed} tokens but the shared tape prefix holds {} (memra #657)",
            tokens.len()
        );
        tokens
    }

    /// The whole tape, for a mode that samples past the shared prefix. Only the pinned tape has
    /// the bytes such a mode was calibrated on, so the rebuild is refused.
    pub fn full_pinned(&self) -> &str {
        assert_eq!(
            self.sha256, PINNED_SHA256,
            "this mode reads past the {SHARED_PREFIX_BYTES}-byte shared prefix and needs the \
             pinned tape (memra #657)"
        );
        &self.text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_an_unknown_tape_and_cuts_on_a_char_boundary() {
        let dir = std::env::temp_dir().join(format!("dsv4-tape-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tape.txt");
        std::fs::write(&path, "not the tape").unwrap();
        let err = SourceTape::read(&path).err().unwrap();
        assert!(err.contains("#657"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();

        let mut text = "a".repeat(SHARED_PREFIX_BYTES - 1);
        text.push('\u{00e9}');
        let tape = SourceTape {
            text,
            sha256: REBUILD_SHA256,
        };
        assert_eq!(tape.shared().len(), SHARED_PREFIX_BYTES - 1);
    }

    #[test]
    #[should_panic(expected = "needs the pinned tape")]
    fn full_reads_refuse_the_rebuild() {
        let tape = SourceTape {
            text: String::new(),
            sha256: REBUILD_SHA256,
        };
        tape.full_pinned();
    }
}
