//! CPU reference schedules. These do not qualify any production backend or device.
pub use memra_tier::conformance;
mod peer;
mod rows;
mod services;
mod support;
mod transfer;
mod wire;

mod v12_bindings;

mod v13_bindings;

mod resident_bindings;

mod fault_arm_bindings;

mod reader_fence_bindings;

mod d2d_capture_bindings;

mod d2d_restore_bindings;

// Day 22 (WP-A, memra#536 Move 2 slice 3): the receipt term of both D2D classes.
mod d2d_receipt_bindings;

// Day 30 (WP-A, memra#536 Move 2 owed item 1): the typed f32 spans of a demote batch.
mod d2h_span_bindings;

// Day 32 (WP-A, memra#536 Move 2 owed item 1, the H2D half): the typed f32 spans of a promote batch.
mod h2d_span_bindings;
