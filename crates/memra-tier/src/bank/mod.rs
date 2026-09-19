//! Day-1 CPU prototype, not wired into the engine or the shared contracts crate.
//! All contract definitions here are proposals pending the lead's interface freeze.
//! No CUDA, pinning, UVA or actual NVMe implementation is claimed by this module.
#![forbid(unsafe_code)]

mod residency;
mod rows;
mod types;

pub use residency::*;
pub use rows::*;
pub use types::*;
