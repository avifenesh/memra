//! Versioned JSONL per-operation deltas, not cumulative counters.
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StorageSample {
    pub schema_version: u32,
    pub fixture: String,
    pub backend_requested: String,
    pub backend_actual: String,
    pub status: String,
    pub valid_bytes: u64,
    pub padded_bytes: u64,
    pub io_bytes: u64,
    /// None for buffered filesystem/CPU fake: submitted bytes are not device traffic.
    pub physical_bytes: Option<u64>,
    pub queue_ns: Option<u64>,
    pub io_ns: Option<u64>,
    pub h2d_ns: Option<u64>,
    pub d2h_ns: Option<u64>,
    pub total_ns: u64,
    pub inflight: u32,
    pub pinned_bytes: u64,
    /// None when RSS/backing allocation was not instrumented (never a guessed peak).
    pub pageable_bytes: Option<u64>,
    pub fallbacks: u64,
    pub payload_sha256: String,
}
impl StorageSample {
    pub fn json_line(&self) -> serde_json::Result<String> {
        serde_json::to_string(self)
    }
}
