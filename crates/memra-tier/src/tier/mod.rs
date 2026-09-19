//! B-owned shared admission control. Inject ONE governor into all tier consumers.
pub mod governor;
pub use governor::{Governor, QueueOutcome};
