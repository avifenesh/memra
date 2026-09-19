//! Frozen CPU contracts, not a storage implementation or a GPU qualification.
//!
//! Lane merges will add `object_store`, `io`, `pool`, `telemetry` (WP-A),
//! `tier` (WP-B), `bank` (WP-C), and `peer`, `placement` (WP-D).
//! No empty shadow modules: implementations import this single contract namespace.
#![forbid(unsafe_code)]
pub mod contracts;
pub mod peer;
pub mod placement;
