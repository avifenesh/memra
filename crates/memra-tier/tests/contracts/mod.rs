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
