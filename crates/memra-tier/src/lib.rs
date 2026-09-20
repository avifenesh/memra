//! Frozen CPU contracts, not a storage implementation or a GPU qualification.
//!
//! Lane merges will add `object_store`, `io`, `pool`, `telemetry` (WP-A),
//! `tier` (WP-B), `bank` (WP-C), and `peer`, `placement` (WP-D).
//! No empty shadow modules: implementations import this single contract namespace.
#![forbid(unsafe_code)]
pub mod bank;
pub mod contracts;

/// Conformance schedules every backend runs: the CPU contract tests (fake backends) and the
/// native qualification gates (`tier-transfer-gate`) execute these exact functions.
pub mod conformance;
pub mod io;
pub mod object_store;
pub mod peer;
pub mod placement;
pub mod pool;
pub mod telemetry;
pub mod tier;
