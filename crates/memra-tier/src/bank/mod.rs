//! Exact-byte CPU host bank/row implementation of the frozen shared contracts.
//! No CUDA, pinning, UVA, real NVMe or runtime dispatch qualification.
#![forbid(unsafe_code)]
mod adapters;
mod residency;
mod rows;
mod types;
pub use adapters::*;
pub use residency::*;
pub use rows::*;
pub use types::*;

mod transport;
pub use transport::*;

mod prediction;
pub use prediction::*;
