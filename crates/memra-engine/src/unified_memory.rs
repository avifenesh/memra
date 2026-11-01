//! Unified-memory devices (an integrated GPU sharing system DRAM with the CPU, such as the
//! GB10 in a DGX Spark) and the free-memory reading their sizing decisions use.
//!
//! Detection reads the device property `CU_DEVICE_ATTRIBUTE_INTEGRATED`, never the device
//! name, so every integrated part takes the same branch and every discrete card keeps the
//! plain `cuMemGetInfo` reading byte for byte.
//!
//! Why the reading differs. On a unified-memory device `cuMemGetInfo` reports the system pool
//! but leaves out memory the kernel can reclaim on demand (page cache, SWAP): NVIDIA's DGX
//! Spark known-issues page says the reported size "may be smaller than the actual allocatable
//! memory" and points to `/proc/meminfo` `MemAvailable` instead. After a weight load the page
//! cache holds the checkpoint files, so the raw reading undercounts by roughly the checkpoint
//! size. `MemAvailable` already subtracts the kernel's own watermark reserve, so a floor kept
//! on top of it (the admission driver-headroom rung, the transient reserve) stays a floor.
//!
//! Only SIZING reads (prefix-cache budget, admission's effective free, the MoE residency
//! decision) take [`allocatable_mem_info`]. The crash guards that protect the driver itself
//! (graph-launch floor, pre-capture headroom, the driver-headroom trim) keep the raw reading:
//! on unified memory it can only be lower, so they stay on the conservative side.

use cudarc::driver::{CudaContext, DriverError};

/// Whether the device shares system memory with the host (`CU_DEVICE_ATTRIBUTE_INTEGRATED`).
/// An unreadable attribute reads as discrete, the pre-existing behavior.
pub fn is_unified(ctx: &CudaContext) -> bool {
    use cudarc::driver::sys::CUdevice_attribute_enum as A;
    ctx.attribute(A::CU_DEVICE_ATTRIBUTE_INTEGRATED)
        .is_ok_and(|v| v != 0)
}

/// `MemAvailable` from `/proc/meminfo` content, in bytes. `None` when absent or malformed.
pub fn parse_mem_available_bytes(meminfo: &str) -> Option<usize> {
    let line = meminfo.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kb = line.split_whitespace().nth(1)?.parse::<usize>().ok()?;
    Some(kb.saturating_mul(1024))
}

/// The host's `MemAvailable` now, in bytes.
pub fn host_mem_available_bytes() -> Option<usize> {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| parse_mem_available_bytes(&s))
}

/// Allocatable free bytes on a unified-memory device: the larger of the driver's reading and
/// `MemAvailable`. The larger one, because the driver reading omits reclaimable pages and
/// `MemAvailable` is the kernel's own estimate of what an allocation can get without
/// swapping. An unreadable `/proc/meminfo` keeps the driver reading.
pub fn unified_free_bytes(driver_free: usize, mem_available: Option<usize>) -> usize {
    mem_available.map_or(driver_free, |avail| avail.max(driver_free))
}

/// `(free, total)` for sizing decisions: `cuMemGetInfo` on a discrete card (unchanged), and
/// [`unified_free_bytes`] on a unified-memory device.
pub fn allocatable_mem_info(ctx: &CudaContext) -> Result<(usize, usize), DriverError> {
    let (free, total) = ctx.mem_get_info()?;
    if is_unified(ctx) {
        Ok((unified_free_bytes(free, host_mem_available_bytes()), total))
    } else {
        Ok((free, total))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mem_available_is_parsed_in_bytes_and_only_from_its_own_line() {
        let meminfo = "MemTotal:       127535104 kB\nMemFree:         4000000 kB\n\
                       MemAvailable:   98000000 kB\nBuffers:            1024 kB\n";
        assert_eq!(
            parse_mem_available_bytes(meminfo),
            Some(98_000_000usize * 1024)
        );
        assert_eq!(parse_mem_available_bytes("MemFree: 12 kB\n"), None);
        assert_eq!(parse_mem_available_bytes("MemAvailable: junk kB\n"), None);
    }

    #[test]
    fn unified_free_takes_the_reclaimable_reading_and_never_lowers_the_driver_one() {
        // The Spark shape: the driver reports ~70 GB while the page cache holds the checkpoint.
        let driver = 70usize << 30;
        assert_eq!(unified_free_bytes(driver, Some(95 << 30)), 95 << 30);
        // A driver reading above MemAvailable is kept: the larger reading is the allocatable one.
        assert_eq!(unified_free_bytes(driver, Some(60 << 30)), driver);
        // Unreadable meminfo keeps the pre-existing reading.
        assert_eq!(unified_free_bytes(driver, None), driver);
    }
}
