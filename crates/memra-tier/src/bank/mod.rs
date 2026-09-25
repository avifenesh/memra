//! Exact-byte CPU host bank/row implementation of the frozen shared contracts.
//! No CUDA, pinning, UVA, real NVMe or runtime dispatch qualification.
#![forbid(unsafe_code)]
mod adapters;
mod device_rows;
pub use device_rows::*;
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

mod slru;
pub use slru::*;
mod source;
pub use source::*;

mod expert_dispatch;
pub use expert_dispatch::*;

mod owner_proxy;
pub use owner_proxy::*;

mod host_buffer;
pub use host_buffer::*;
