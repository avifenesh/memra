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
//! SIZING reads (prefix-cache budget, admission's effective free, the MoE residency decision),
//! the pre-capture headroom guard and admission's driver-headroom rung take
//! [`allocatable_mem_info`]. The guard moved in memra#1221: with a full page cache the raw
//! reading sat at 2-3 GB while `MemAvailable` was near 100 GB, so the guard trimmed the pool (a
//! 260k prompt's publication then grew it back on the scheduler thread) and refused 1k requests'
//! captures, while a capture with the trim skipped at that reading succeeded. The rung moved in
//! memra#1257 for the same reason: at that reading it trimmed the pool before nearly every prime.
//! The pooled MTP verify's memory cap (memra#1266) reads it, and so does the pooled DFlash2
//! verify's width (memra#1307).
//! The graph-launch floor keeps the raw reading: on unified memory it can only be lower, so it
//! stays on the conservative side. A capture admitted on `MemAvailable` therefore still runs
//! eagerly in any round where the raw reading falls under the 256 MiB launch floor (memra#1221's
//! cells kept it at 1.1 GB or more). A failed capture's appetite floor keeps the raw entry
//! reading for the same reason: it is at most what the capture could get, so the bound it records
//! in the draft-state high-water is proven. The DeepSeek-V4 route's memory door reads the raw
//! value: the route loads only on discrete cards (`dsv4_gpu.rs` `stage_memory`).
//!
//! A DELTA is not a level. The per-session draft-state measurement subtracts two readings taken
//! around a capture, and on unified memory neither the raw reading nor `MemAvailable` holds still
//! while the kernel moves pages between the page cache and free memory (memra#1307's probe on
//! GB10, a full page cache: a 256 MiB allocation read -500 to +308 MiB on the raw reading and
//! -368 to +878 MiB on `MemAvailable`). The raw reading is `MemFree`: page cache filling lowers it
//! and reclaim raises it. `MemAvailable` estimates the reclaimable cache from the kernel's LRU
//! lists and missed the allocation by up to 620 MiB either way while reclaim shrank `Cached`.
//! Free memory plus what reclaim turns into free memory (`MemFree + Buffers + Cached +
//! SReclaimable`, one read of `/proc/meminfo`) moves by the allocation alone, since filling and
//! reclaim move pages between its terms: 251-263 MiB in the same 256 MiB samples. That is
//! [`delta_free_bytes`], which the measurement reads at both ends. A level never takes it: not
//! every cached page is reclaimable.

use cudarc::driver::{CudaContext, DriverError};

/// Whether the device shares system memory with the host (`CU_DEVICE_ATTRIBUTE_INTEGRATED`).
/// An unreadable attribute reads as discrete, the pre-existing behavior.
pub fn is_unified(ctx: &CudaContext) -> bool {
    use cudarc::driver::sys::CUdevice_attribute_enum as A;
    ctx.attribute(A::CU_DEVICE_ATTRIBUTE_INTEGRATED)
        .is_ok_and(|v| v != 0)
}

/// One `/proc/meminfo` field (`name` with its colon) from the file's content, in bytes. `None`
/// when absent or malformed.
fn meminfo_field_bytes(meminfo: &str, name: &str) -> Option<usize> {
    let line = meminfo.lines().find(|l| l.starts_with(name))?;
    let kb = line.split_whitespace().nth(1)?.parse::<usize>().ok()?;
    Some(kb.saturating_mul(1024))
}

/// `MemAvailable` from `/proc/meminfo` content, in bytes. `None` when absent or malformed.
pub fn parse_mem_available_bytes(meminfo: &str) -> Option<usize> {
    meminfo_field_bytes(meminfo, "MemAvailable:")
}

/// Free memory plus what reclaim turns into free memory, from `/proc/meminfo` content, in bytes:
/// `MemFree + Buffers + Cached + SReclaimable`, from one read. `None` when a field is absent or
/// malformed.
pub fn parse_free_plus_reclaimable_bytes(meminfo: &str) -> Option<usize> {
    ["MemFree:", "Buffers:", "Cached:", "SReclaimable:"]
        .iter()
        .try_fold(0usize, |sum, name| {
            Some(sum.saturating_add(meminfo_field_bytes(meminfo, name)?))
        })
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

/// The reading a free-memory DELTA takes (memra#1307), given the `cuMemGetInfo` free reading
/// `driver_free` taken with it: that reading on a discrete card, unchanged, and on unified memory
/// [`parse_free_plus_reclaimable_bytes`] (see the module notes). `None` on unified memory when
/// `/proc/meminfo` cannot be read, so a delta is never taken across two different readings.
pub fn delta_free_bytes(ctx: &CudaContext, driver_free: usize) -> Option<usize> {
    if is_unified(ctx) {
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|s| parse_free_plus_reclaimable_bytes(&s))
    } else {
        Some(driver_free)
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

    /// memra#1307: the delta reading sums free memory, the page cache and the reclaimable slab from
    /// one read; `SwapCached` is not `Cached`, and a missing field reads as no reading.
    #[test]
    fn free_plus_reclaimable_reads_each_field_from_its_own_line() {
        let meminfo = "MemTotal:       127535104 kB\nMemFree:         2000000 kB\n\
                       MemAvailable:   98000000 kB\nBuffers:            1024 kB\n\
                       Cached:         95000000 kB\nSwapCached:            7 kB\n\
                       SReclaimable:     900000 kB\nSUnreclaim:       300000 kB\n";
        assert_eq!(
            parse_free_plus_reclaimable_bytes(meminfo),
            Some((2_000_000usize + 1_024 + 95_000_000 + 900_000) * 1024)
        );
        assert_eq!(parse_free_plus_reclaimable_bytes("MemFree: 12 kB\n"), None);
        let no_cached = "MemFree: 12 kB\nBuffers: 1 kB\nSwapCached: 7 kB\nSReclaimable: 3 kB\n";
        assert_eq!(parse_free_plus_reclaimable_bytes(no_cached), None);
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
