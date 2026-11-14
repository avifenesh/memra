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
//! around a capture and keeps the result as a high-water, and on unified memory neither the raw
//! reading nor `MemAvailable` holds still while the kernel moves pages between the page cache and
//! free memory (memra#1307's probe on GB10, a full page cache: a 256 MiB allocation read -500 to
//! +308 MiB on the raw reading and -368 to +878 MiB on `MemAvailable`). The raw reading is
//! `MemFree`: page cache filling lowers it and reclaim raises it. `MemAvailable` estimates the
//! reclaimable cache from the kernel's LRU lists and missed the allocation by up to 620 MiB either
//! way while reclaim shrank `Cached`; across one served boot bracket that parks 34 MiB it fell
//! 433 MiB. Free memory plus what reclaim turns into free memory (`MemFree + Buffers + Cached +
//! SReclaimable`, one read of `/proc/meminfo`) cancels both, since filling and reclaim move pages
//! between its terms: 251-263 MiB in the same 256 MiB samples, and 34 and 35 MB recorded in boots
//! under a filling page cache where the raw reading recorded 46 and 55. That is
//! [`delta_free_bytes`], which the measurement reads at both ends. It is still a host-wide
//! reading, and with reclaim running it has read low: 3 of 13 boots that met a full page cache
//! read 2, 18 and 22 MiB for the same capture (`docs/decisions/UNIFIED-MEMORY-READERS.md`). A
//! level never takes it: not every cached page is reclaimable.

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

/// The reading a free-memory DELTA takes (memra#1307), from its inputs: the `cuMemGetInfo` free
/// reading `driver_free` on a discrete card, unchanged and whatever `meminfo` holds, and on
/// unified memory [`parse_free_plus_reclaimable_bytes`] of `/proc/meminfo`'s content (see the
/// module notes). `None` on unified memory when the content is missing or malformed: a delta is
/// then not taken at all, never across two different readings.
pub fn delta_free_from(unified: bool, driver_free: usize, meminfo: Option<&str>) -> Option<usize> {
    if unified {
        meminfo.and_then(parse_free_plus_reclaimable_bytes)
    } else {
        Some(driver_free)
    }
}

/// [`delta_free_from`] on the host's `/proc/meminfo`, which only a unified-memory device reads.
/// `unified` is the engine's own answer (`Engine::unified_memory`), read once when it was built,
/// so both ends of a delta take the same reading.
pub fn delta_free_bytes(unified: bool, driver_free: usize) -> Option<usize> {
    let meminfo = unified
        .then(|| std::fs::read_to_string("/proc/meminfo").ok())
        .flatten();
    delta_free_from(unified, driver_free, meminfo.as_deref())
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

    /// memra#1307: the delta reading, as a table. A discrete card reads the driver's free memory
    /// whatever `/proc/meminfo` holds; unified memory reads free plus reclaimable, which is
    /// neither the driver's reading nor `MemAvailable`, and reads nothing when the file is
    /// missing or short of a field.
    #[test]
    fn delta_reading_is_the_driver_free_on_a_discrete_card_and_free_plus_reclaimable_on_unified() {
        // One of the GB10 probe's samples: MemFree 1.0 GB, MemAvailable 115.3 GiB, 115.4 GiB cached.
        let meminfo = "MemFree: 1003827 kB\nMemAvailable: 120910950 kB\nBuffers: 3891 kB\n\
                       Cached: 121012122 kB\nSReclaimable: 734413 kB\n";
        let fpr = (1_003_827usize + 3_891 + 121_012_122 + 734_413) * 1024;
        let driver = 1_003_827usize * 1024;
        let rows: [(bool, usize, Option<&str>, Option<usize>); 6] = [
            (false, driver, Some(meminfo), Some(driver)),
            (false, driver, None, Some(driver)),
            (false, 0, Some("junk"), Some(0)),
            (true, driver, Some(meminfo), Some(fpr)),
            (true, driver, None, None),
            (true, driver, Some("MemFree: 12 kB\nCached: 7 kB\n"), None),
        ];
        for (unified, driver_free, content, want) in rows {
            assert_eq!(
                delta_free_from(unified, driver_free, content),
                want,
                "unified={unified} driver_free={driver_free} content={content:?}"
            );
        }
        // The unified reading is not the level's: `MemAvailable` here is 1.8 GiB lower.
        assert_ne!(Some(fpr), parse_mem_available_bytes(meminfo));
        assert_ne!(
            fpr,
            unified_free_bytes(driver, parse_mem_available_bytes(meminfo))
        );

        // Wiring. `delta_free_bytes` only feeds `delta_free_from` the host's file and never asks
        // the device; the engine reads "unified?" once, when it is built, and hands back that
        // answer.
        let src = include_str!("unified_memory.rs");
        let production = &src[..src.find("#[cfg(test)]").unwrap()];
        let at = production
            .find("pub fn delta_free_bytes(unified: bool, driver_free: usize)")
            .expect("delta_free_bytes takes the engine's answer");
        let body = &production[at..];
        let body = &body[..body.find("\n}\n").unwrap()];
        assert!(
            body.contains("delta_free_from(unified, driver_free, meminfo.as_deref())")
                && !body.contains("is_unified(")
                && !body.contains("mem_available"),
            "{body}"
        );
        let engine = include_str!("lib.rs");
        assert_eq!(
            engine
                .matches("let unified_memory = crate::unified_memory::is_unified(&gpu.ctx);")
                .count(),
            1
        );
        assert!(engine.contains(
            "pub fn unified_memory(&self) -> bool {\n        self.unified_memory\n    }"
        ));
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
