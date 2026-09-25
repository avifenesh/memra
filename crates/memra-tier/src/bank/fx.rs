//! Day 63 (I13 change 4) and day 64 (I14, `research/spill-c-20260919/DAY64.md`): the bank's deterministic hasher for
//! its maps keyed by catalog records (the SLRU's two maps, the catalog's index, the host cache). Keys are catalog
//! records, not untrusted input, and no user of these maps depends on their iteration order.
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// Day 63 (I13 change 4, `research/spill-c-20260919/DAY63.md`): the rustc Fx step (rotate left 5, xor, multiply by
/// `0x517cc1b727220a95`) over a key's hash writes, eight bytes at a time. Deterministic and cheap for the SLRU's two
/// maps, whose keys are catalog records (not untrusted input) and which are never iterated, so no order is observable.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FxHasher {
    hash: u64,
}
const FX_SEED: u64 = 0x517c_c1b7_2722_0a95;
impl FxHasher {
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(FX_SEED);
    }
}
impl Hasher for FxHasher {
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            self.add(u64::from_le_bytes(chunk.try_into().expect("eight bytes")));
        }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            let mut word = [0u8; 8];
            word[..rest.len()].copy_from_slice(rest);
            self.add(u64::from_le_bytes(word));
        }
    }
    fn write_u8(&mut self, i: u8) {
        self.add(u64::from(i));
    }
    fn write_u16(&mut self, i: u16) {
        self.add(u64::from(i));
    }
    fn write_u32(&mut self, i: u32) {
        self.add(u64::from(i));
    }
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }
    fn finish(&self) -> u64 {
        self.hash
    }
}
pub(crate) type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;
