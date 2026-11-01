//! The serving device's shape, published once by the worker before it reports ready, so
//! HTTP-layer defaults that depend on the hardware (the streaming first-token deadline and its
//! SSE prefill commit on unified-memory devices) read the device instead of an env flag.
//!
//! The server binds its socket only after the worker's ready verdict, so every request sees the
//! published value. Before publication (unit tests, a CPU-only process) nothing is published and
//! every reader keeps the discrete-card defaults.

use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ServingDevice {
    /// `CU_DEVICE_ATTRIBUTE_INTEGRATED`: the GPU shares system DRAM (GB10, DGX Spark).
    pub(crate) unified_memory: bool,
    /// `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT` of the primary device.
    pub(crate) sm_count: u32,
    /// The largest served context across the loaded models (`resolve_env_ctx`), in tokens.
    pub(crate) max_served_ctx: usize,
}

static DEVICE: OnceLock<ServingDevice> = OnceLock::new();

/// First publication wins: a worker respawn runs on the same device and context.
pub(crate) fn publish(device: ServingDevice) {
    let _ = DEVICE.set(device);
}

pub(crate) fn get() -> Option<&'static ServingDevice> {
    DEVICE.get()
}
