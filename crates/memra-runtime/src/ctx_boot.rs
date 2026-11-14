//! CUDA context creation, with the failing call named and, on a unified-memory device, a bounded
//! retry of an out-of-memory answer the host's reclaimable memory contradicts (memra#1316).
//!
//! Why a retry, and why only here. Creating a context makes the driver allocate the new
//! channel's context buffers. On a unified-memory device (the GB10 in a DGX Spark) those come
//! from system memory as multi-page blocks, through the kernel's page allocator with
//! `__GFP_RETRY_MAYFAIL` (`nv_compute_gfp_mask` in the driver's `nv-vm.c`): the kernel reclaims
//! and compacts for a while and then fails the allocation instead of waiting. With the page
//! cache full and other tasks taking pages as fast as the kernel frees them, a block of that
//! size may not exist at that moment, and `cuDevicePrimaryCtxRetain` returns
//! `CUDA_ERROR_OUT_OF_MEMORY` while `MemAvailable` reads over 100 GB. The driver's own
//! big-page allocations fall back to small pages in that state; the context buffer has no such
//! fallback. Nothing the process holds is involved (it holds nothing yet), so the same call
//! succeeds once the kernel has freed a block: a retry is the fix, not a workaround.
//!
//! The retry is refused where the answer can be true: on a discrete card (the answer is about
//! the card's own memory, which no wait frees), and on a unified-memory device whose
//! `MemAvailable` is under [`ReclaimRetry::min_available`] or unreadable (nothing contradicts
//! the driver).
//!
//! [`retry_reclaim_race`] is the policy as a pure function over injected closures, so a CPU test
//! pins it with a fake allocator.

use cudarc::driver::{CudaContext, DriverError, result, sys};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The host memory lines a boot failure reports, in bytes, and the count of free blocks of
/// order 9 and up in the Normal zone (`/proc/buddyinfo`; 2 MiB and larger with 4 KiB pages, the
/// size of the context buffer on GB10): the reading that tells "no large block right now" from
/// "no memory".
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HostMem {
    pub free: usize,
    pub available: usize,
    pub cached: usize,
    pub large_free_blocks: Option<usize>,
}

impl std::fmt::Display for HostMem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "host MemFree {} MiB, MemAvailable {} MiB, Cached {} MiB",
            self.free >> 20,
            self.available >> 20,
            self.cached >> 20
        )?;
        if let Some(n) = self.large_free_blocks {
            write!(
                f,
                ", free blocks of order 9 and up (2 MiB with 4 KiB pages): {n}"
            )?;
        }
        Ok(())
    }
}

fn meminfo_bytes(meminfo: &str, key: &str) -> Option<usize> {
    let line = meminfo.lines().find(|l| {
        l.strip_prefix(key)
            .is_some_and(|rest| rest.starts_with(':'))
    })?;
    let kb = line.split_whitespace().nth(1)?.parse::<usize>().ok()?;
    Some(kb.saturating_mul(1024))
}

/// Free blocks of order 9 and up in the Normal zone(s) of `/proc/buddyinfo` content.
pub fn parse_large_free_blocks(buddyinfo: &str) -> Option<usize> {
    let mut total = None;
    for line in buddyinfo.lines() {
        let mut parts = line.split_whitespace();
        if parts.by_ref().nth(3) != Some("Normal") {
            continue;
        }
        let counts: Vec<usize> = parts.filter_map(|p| p.parse().ok()).collect();
        if counts.len() > 9 {
            *total.get_or_insert(0) += counts[9..].iter().sum::<usize>();
        }
    }
    total
}

/// [`HostMem`] from `/proc/meminfo` content (and `/proc/buddyinfo` content when readable).
/// `None` when `MemAvailable` is absent or malformed: the policy then has no host reading.
pub fn parse_host_mem(meminfo: &str, buddyinfo: Option<&str>) -> Option<HostMem> {
    Some(HostMem {
        free: meminfo_bytes(meminfo, "MemFree").unwrap_or(0),
        available: meminfo_bytes(meminfo, "MemAvailable")?,
        cached: meminfo_bytes(meminfo, "Cached").unwrap_or(0),
        large_free_blocks: buddyinfo.and_then(parse_large_free_blocks),
    })
}

/// The host's memory now.
pub fn host_mem() -> Option<HostMem> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let buddy = std::fs::read_to_string("/proc/buddyinfo").ok();
    parse_host_mem(&meminfo, buddy.as_deref())
}

/// The bounded retry of an out-of-memory answer on unified memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReclaimRetry {
    /// The wait before the second try; each later wait doubles, up to `max_backoff`.
    pub first_backoff: Duration,
    pub max_backoff: Duration,
    /// Elapsed time from the start of the first try, tries included: a try whose wait would end
    /// past it is not made. A try the kernel stalls in still counts (a failed creation took 22 to
    /// 266 ms on GB10 and a successful one 3.6 s once), so the bound is on the boot's time, not
    /// on the sleeps; the last try can run past it by its own duration.
    pub budget: Duration,
    /// `MemAvailable` under which an out-of-memory answer is taken as true and not retried.
    pub min_available: usize,
}

/// The serving policy. The numbers come from `research/gb10-boot-oom-20261114/`: see
/// `docs/decisions/GB10-CONTEXT-CREATE-RETRY.md`.
pub const RECLAIM_RETRY: ReclaimRetry = ReclaimRetry {
    first_backoff: Duration::from_millis(50),
    max_backoff: Duration::from_millis(800),
    budget: Duration::from_secs(90),
    min_available: 256 << 20,
};

/// What [`retry_reclaim_race`] reports to its `note` closure. `elapsed` is the time since the
/// first try began.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetryNote {
    /// An out-of-memory answer the policy will retry after `wait`.
    Retrying {
        failed_tries: u32,
        elapsed: Duration,
        wait: Duration,
        host: HostMem,
    },
    /// The call succeeded after at least one retry.
    Recovered { tries: u32, elapsed: Duration },
    /// The budget ran out; the last out-of-memory answer is returned.
    GaveUp {
        tries: u32,
        elapsed: Duration,
        host: Option<HostMem>,
    },
    /// Out of memory and not retried: the host reading says the memory is not there (or
    /// there is no host reading).
    NotReclaimable { host: Option<HostMem> },
}

/// Run `call`; on a unified-memory device, repeat it while it answers out of memory although the
/// host reports at least `policy.min_available` reclaimable, waiting `first_backoff`, doubling
/// to `max_backoff`, until `policy.budget` has elapsed since the first try began.
///
/// Returned unchanged, after one call: every success, every error that is not out of memory, and
/// every error on a discrete device (`unified == false`). `host`, `wait` and `now` (a monotonic
/// clock, any origin) are injected so a test runs the policy without a device or real time.
#[allow(clippy::too_many_arguments)]
pub fn retry_reclaim_race<T, E>(
    policy: &ReclaimRetry,
    unified: bool,
    mut call: impl FnMut() -> Result<T, E>,
    is_oom: impl Fn(&E) -> bool,
    mut host: impl FnMut() -> Option<HostMem>,
    mut wait: impl FnMut(Duration),
    mut now: impl FnMut() -> Duration,
    mut note: impl FnMut(RetryNote),
) -> Result<T, E> {
    let started = now();
    let mut err = match call() {
        Ok(v) => return Ok(v),
        Err(e) => e,
    };
    if !unified || !is_oom(&err) {
        return Err(err);
    }
    let mut tries = 1u32;
    // At least 1 ms a wait, so a policy with a zero backoff still reaches its budget.
    let mut backoff = policy.first_backoff.max(Duration::from_millis(1));
    loop {
        let reading = host();
        let Some(h) = reading.filter(|h| h.available >= policy.min_available) else {
            note(RetryNote::NotReclaimable { host: reading });
            return Err(err);
        };
        let elapsed = now().saturating_sub(started);
        if elapsed + backoff > policy.budget {
            note(RetryNote::GaveUp {
                tries,
                elapsed,
                host: reading,
            });
            return Err(err);
        }
        note(RetryNote::Retrying {
            failed_tries: tries,
            elapsed,
            wait: backoff,
            host: h,
        });
        wait(backoff);
        backoff = (backoff * 2).min(policy.max_backoff.max(Duration::from_millis(1)));
        tries += 1;
        match call() {
            Ok(v) => {
                note(RetryNote::Recovered {
                    tries,
                    elapsed: now().saturating_sub(started),
                });
                return Ok(v);
            }
            Err(e) if is_oom(&e) => err = e,
            Err(e) => return Err(e),
        }
    }
}

/// A boot step that failed, with the driver call named.
#[derive(Debug)]
pub struct BootStepError {
    pub ordinal: usize,
    /// The driver call and what it is for, e.g. "cuDevicePrimaryCtxRetain (context creation)".
    pub call: &'static str,
    pub source: Box<dyn std::error::Error>,
    /// Host memory at the failure, read for out-of-memory answers.
    pub host: Option<HostMem>,
}

impl std::fmt::Display for BootStepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} on device {} failed: {}",
            self.call, self.ordinal, self.source
        )?;
        if let Some(h) = self.host {
            write!(f, " ({h})")?;
        }
        Ok(())
    }
}

impl std::error::Error for BootStepError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.source.as_ref())
    }
}

fn is_driver_oom(err: &DriverError) -> bool {
    err.0 == sys::cudaError_enum::CUDA_ERROR_OUT_OF_MEMORY
}

/// Name the driver call a boot step failed in. On a unified-memory device an out-of-memory
/// answer also carries the host's memory lines, since they are what the answer has to be read
/// against; on a discrete card they say nothing about the card and are left out.
pub fn boot_step<T, E>(
    ordinal: usize,
    call: &'static str,
    result: Result<T, E>,
) -> Result<T, Box<dyn std::error::Error>>
where
    E: std::error::Error + 'static,
{
    result.map_err(|e| {
        let oom = (&e as &dyn std::error::Error)
            .downcast_ref::<DriverError>()
            .is_some_and(is_driver_oom);
        Box::new(BootStepError {
            ordinal,
            call,
            source: Box::new(e),
            host: if oom && ordinal_is_integrated(ordinal) {
                host_mem()
            } else {
                None
            },
        }) as Box<dyn std::error::Error>
    })
}

/// What the `CU_DEVICE_ATTRIBUTE_INTEGRATED` read says: unified only when the driver answers
/// nonzero. An unreadable attribute reads as discrete: the one-call behavior.
fn attribute_says_integrated(attr: Result<i32, DriverError>) -> bool {
    attr.is_ok_and(|v| v != 0)
}

fn read_integrated(device: sys::CUdevice) -> Result<i32, DriverError> {
    unsafe {
        result::device::get_attribute(
            device,
            sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_INTEGRATED,
        )
    }
}

/// Whether `device` shares system memory with the host (`CU_DEVICE_ATTRIBUTE_INTEGRATED`), read
/// without a context. (`memra-engine`'s `unified_memory::is_unified` reads the same attribute
/// once a context exists.)
pub fn device_is_integrated(device: sys::CUdevice) -> bool {
    attribute_says_integrated(read_integrated(device))
}

/// [`device_is_integrated`] for an ordinal, on an error path where the driver is already
/// initialized. A device that cannot be opened reads as discrete.
fn ordinal_is_integrated(ordinal: usize) -> bool {
    result::device::get(ordinal as i32).is_ok_and(device_is_integrated)
}

/// The driver's out-of-memory code as a number, for the retry line. The lines of a boot that
/// recovers must not carry the phrases the serving gates scan a server log for (the code's name
/// and its text), or a boot the retry saved would fail them. The give-up line, the not-retried
/// lines and the boot's own error keep both.
const OOM_CODE: u32 = sys::cudaError_enum::CUDA_ERROR_OUT_OF_MEMORY as u32;

/// The log line of one [`RetryNote`].
pub fn retry_line(ordinal: usize, note: &RetryNote) -> String {
    match *note {
        RetryNote::Retrying {
            failed_tries,
            elapsed,
            wait,
            host,
        } => format!(
            "[ctx-retry] device {ordinal}: context creation returned driver code {OOM_CODE} (no \
             memory) on unified memory (try {failed_tries}, {} ms since the first try began; \
             {host}); trying again in {} ms",
            elapsed.as_millis(),
            wait.as_millis()
        ),
        RetryNote::Recovered { tries, elapsed } => format!(
            "[ctx-retry] device {ordinal}: context created on try {tries}, after {} ms",
            elapsed.as_millis()
        ),
        RetryNote::GaveUp {
            tries,
            elapsed,
            host,
        } => format!(
            "[ctx-retry] device {ordinal}: context creation still out of memory after {tries} \
             tries and {} ms{}; giving up",
            elapsed.as_millis(),
            host.map(|h| format!(" ({h})")).unwrap_or_default()
        ),
        RetryNote::NotReclaimable { host: Some(h) } => format!(
            "[ctx-retry] device {ordinal}: context creation answered out of memory and the host \
             has under {} MiB available ({h}); not retried",
            RECLAIM_RETRY.min_available >> 20
        ),
        RetryNote::NotReclaimable { host: None } => format!(
            "[ctx-retry] device {ordinal}: context creation answered out of memory and the \
             host's MemAvailable could not be read; not retried"
        ),
    }
}

/// The wiring of [`create_context`]: the serving policy, the device's attribute read deciding
/// "unified", and the driver's out-of-memory test, around a creator. Everything that touches the
/// driver, the host or time is passed in, so CPU tests hold each piece to its place.
fn create_with<T>(
    integrated: Result<i32, DriverError>,
    create: impl FnMut() -> Result<T, DriverError>,
    host: impl FnMut() -> Option<HostMem>,
    wait: impl FnMut(Duration),
    now: impl FnMut() -> Duration,
    note: impl FnMut(RetryNote),
) -> Result<T, DriverError> {
    retry_reclaim_race(
        &RECLAIM_RETRY,
        attribute_says_integrated(integrated),
        create,
        is_driver_oom,
        host,
        wait,
        now,
        note,
    )
}

/// `CudaContext::new` with the failing call named, and [`RECLAIM_RETRY`] on a unified-memory
/// device.
///
/// `CudaContext::new` is `cuInit`, `cuDeviceGet`, `cuDevicePrimaryCtxRetain`, one device attribute
/// and `cuCtxSetCurrent`. The first two run here first, under their own names (a missing device
/// must not read as a failed context creation), so what the constructor can still fail in is the
/// retain, the attribute and the bind; the retain is the one that allocates, and its name is the
/// label. A failed retain leaves nothing behind, so the whole constructor is what gets repeated.
pub fn create_context(ordinal: usize) -> Result<Arc<CudaContext>, Box<dyn std::error::Error>> {
    boot_step(ordinal, "cuInit", result::init())?;
    let device = boot_step(ordinal, "cuDeviceGet", result::device::get(ordinal as i32))?;
    let clock = Instant::now();
    boot_step(
        ordinal,
        "cuDevicePrimaryCtxRetain (context creation)",
        create_with(
            read_integrated(device),
            || CudaContext::new(ordinal),
            host_mem,
            std::thread::sleep,
            || clock.elapsed(),
            |note| eprintln!("{}", retry_line(ordinal, &note)),
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};

    const OOM: &str = "out of memory";
    const SPARK_FULL_CACHE: HostMem = HostMem {
        free: 900 << 20,
        available: 119 << 30,
        cached: 118 << 30,
        large_free_blocks: Some(0),
    };
    const MS: fn(u64) -> Duration = Duration::from_millis;

    /// A fake allocator that answers out of memory `fail` times and then succeeds, on a fake
    /// clock: every call takes `call_cost` and every wait its own length. With the calls, the
    /// waits, the notes and the clock's reading at the end.
    struct Run {
        result: Result<u32, &'static str>,
        calls: u32,
        waits: Vec<Duration>,
        notes: Vec<RetryNote>,
        clock: Duration,
    }

    fn run(
        policy: &ReclaimRetry,
        unified: bool,
        fail: u32,
        host: Option<HostMem>,
        call_cost: Duration,
    ) -> Run {
        let calls = Cell::new(0u32);
        // The clock starts away from zero: only differences may matter.
        let clock = Cell::new(Duration::from_secs(1000));
        let waits = RefCell::new(Vec::new());
        let notes = RefCell::new(Vec::new());
        let result = retry_reclaim_race(
            policy,
            unified,
            || {
                calls.set(calls.get() + 1);
                clock.set(clock.get() + call_cost);
                if calls.get() <= fail {
                    Err(OOM)
                } else {
                    Ok(calls.get())
                }
            },
            |e| *e == OOM,
            || host,
            |d| {
                clock.set(clock.get() + d);
                waits.borrow_mut().push(d)
            },
            || clock.get(),
            |n| notes.borrow_mut().push(n),
        );
        Run {
            result,
            calls: calls.get(),
            waits: waits.into_inner(),
            notes: notes.into_inner(),
            clock: clock.get() - Duration::from_secs(1000),
        }
    }

    #[test]
    fn unified_boot_survives_an_out_of_memory_answer_the_page_cache_contradicts() {
        // The GB10 shape of memra#1316: context creation answers out of memory twice while the
        // host has 119 GiB reclaimable, then succeeds. The boot must come up.
        let r = run(&RECLAIM_RETRY, true, 2, Some(SPARK_FULL_CACHE), MS(200));
        assert_eq!(r.result, Ok(3), "the third try's context is the boot's");
        assert_eq!(r.calls, 3);
        assert_eq!(
            r.waits,
            vec![RECLAIM_RETRY.first_backoff, RECLAIM_RETRY.first_backoff * 2],
            "one wait before each retry, doubling"
        );
        // Elapsed time counts the tries: three calls of 200 ms and waits of 50 and 100 ms.
        assert_eq!(
            r.notes.last(),
            Some(&RetryNote::Recovered {
                tries: 3,
                elapsed: MS(750)
            })
        );
    }

    #[test]
    fn discrete_card_keeps_its_first_answer() {
        // A discrete card's out of memory is the card's own memory: one call, no wait, no note.
        let r = run(&RECLAIM_RETRY, false, 1, Some(SPARK_FULL_CACHE), MS(200));
        assert_eq!(r.result, Err(OOM));
        assert_eq!((r.calls, r.waits.len(), r.notes.len()), (1, 0, 0));
    }

    #[test]
    fn a_host_that_is_really_short_is_not_retried() {
        let short = HostMem {
            available: RECLAIM_RETRY.min_available - 1,
            ..SPARK_FULL_CACHE
        };
        let r = run(&RECLAIM_RETRY, true, 1, Some(short), MS(200));
        assert_eq!(r.result, Err(OOM));
        assert_eq!((r.calls, r.waits.len()), (1, 0));
        assert_eq!(
            r.notes,
            vec![RetryNote::NotReclaimable { host: Some(short) }]
        );
        // Exactly at the floor is still retried.
        let at_floor = HostMem {
            available: RECLAIM_RETRY.min_available,
            ..SPARK_FULL_CACHE
        };
        let r = run(&RECLAIM_RETRY, true, 1, Some(at_floor), MS(200));
        assert_eq!((r.result, r.calls), (Ok(2), 2));
        // No host reading: nothing contradicts the driver, so its answer stands.
        let r = run(&RECLAIM_RETRY, true, 1, None, MS(200));
        assert_eq!((r.result, r.calls, r.waits.len()), (Err(OOM), 1, 0));
    }

    #[test]
    fn the_waits_double_to_the_cap_and_stop_at_the_budget() {
        let policy = ReclaimRetry {
            first_backoff: MS(50),
            max_backoff: MS(200),
            budget: MS(950),
            min_available: 1 << 30,
        };
        // Calls of 100 ms. After the fourth, 750 ms have passed and the next wait of 200 ms
        // ends exactly on the budget: it is made. After the fifth, 1,050 ms have passed.
        let r = run(&policy, true, u32::MAX, Some(SPARK_FULL_CACHE), MS(100));
        assert_eq!(r.result, Err(OOM));
        let ms: Vec<u128> = r.waits.iter().map(Duration::as_millis).collect();
        assert_eq!(ms, vec![50, 100, 200, 200]);
        assert_eq!(r.calls, 5);
        assert_eq!(
            r.notes.last(),
            Some(&RetryNote::GaveUp {
                tries: 5,
                elapsed: MS(1050),
                host: Some(SPARK_FULL_CACHE)
            })
        );
        // Each retry is announced with the tries that failed, the time so far and the next wait.
        assert_eq!(
            r.notes[1],
            RetryNote::Retrying {
                failed_tries: 2,
                elapsed: MS(250),
                wait: MS(100),
                host: SPARK_FULL_CACHE
            }
        );
        // The serving policy's own ladder with instant tries: 50, 100, 200, 400, then 800 ms
        // until 90 s have passed.
        let r = run(
            &RECLAIM_RETRY,
            true,
            u32::MAX,
            Some(SPARK_FULL_CACHE),
            MS(0),
        );
        assert_eq!((r.calls, r.waits.len()), (116, 115));
        assert_eq!(r.clock, MS(89_550));
        // A zero backoff cannot spin: every wait is at least 1 ms, so the budget still ends it.
        let zero = ReclaimRetry {
            first_backoff: Duration::ZERO,
            max_backoff: Duration::ZERO,
            budget: MS(5),
            min_available: 1 << 30,
        };
        let r = run(&zero, true, u32::MAX, Some(SPARK_FULL_CACHE), MS(0));
        assert_eq!((r.result, r.calls), (Err(OOM), 6));
    }

    #[test]
    fn the_budget_bounds_elapsed_time_not_the_sleeps() {
        // Tries the kernel stalls in for 3 s each (one successful creation took 3.6 s on GB10).
        // A bound on the sleeps alone would let all 116 tries of the ladder run: 116 tries of
        // 3 s and 89.55 s of sleeps, about 7 minutes. The bound is on the clock: the boot stops
        // after 25 tries, 91.75 s in, having slept 16.75 s.
        let r = run(
            &RECLAIM_RETRY,
            true,
            u32::MAX,
            Some(SPARK_FULL_CACHE),
            MS(3000),
        );
        assert_eq!(r.result, Err(OOM));
        assert_eq!(r.calls, 25);
        assert_eq!(r.waits.iter().sum::<Duration>(), MS(16_750));
        assert_eq!(r.clock, MS(91_750));
        // No try starts after the budget; the last one may end past it by its own length.
        assert!(r.clock <= RECLAIM_RETRY.budget + MS(3000));
        assert_eq!(
            r.notes.last(),
            Some(&RetryNote::GaveUp {
                tries: 25,
                elapsed: MS(91_750),
                host: Some(SPARK_FULL_CACHE)
            })
        );
        // The same stall with a recovery inside the budget: the context of try 20 is the boot's.
        let r = run(&RECLAIM_RETRY, true, 19, Some(SPARK_FULL_CACHE), MS(3000));
        assert_eq!((r.result, r.calls), (Ok(20), 20));
    }

    #[test]
    fn the_host_reading_is_taken_again_before_every_retry() {
        // Reclaimable at the first failure, short at the second: one retry, then the answer stands.
        let reads = Cell::new(0u32);
        let calls = Cell::new(0u32);
        let clock = Cell::new(Duration::ZERO);
        let short = HostMem {
            available: RECLAIM_RETRY.min_available - 1,
            ..SPARK_FULL_CACHE
        };
        let r: Result<(), &str> = retry_reclaim_race(
            &RECLAIM_RETRY,
            true,
            || {
                calls.set(calls.get() + 1);
                Err(OOM)
            },
            |e| *e == OOM,
            || {
                reads.set(reads.get() + 1);
                Some(if reads.get() == 1 {
                    SPARK_FULL_CACHE
                } else {
                    short
                })
            },
            |d| clock.set(clock.get() + d),
            || clock.get(),
            |_| {},
        );
        assert_eq!((r, calls.get(), reads.get()), (Err(OOM), 2, 2));
    }

    #[test]
    fn another_error_ends_the_retry_at_once() {
        // The fake clock moves with the waits in every test, so a policy that wrongly keeps
        // retrying still ends at its budget and the test fails instead of hanging.
        let calls = Cell::new(0u32);
        let clock = Cell::new(Duration::ZERO);
        let r: Result<(), &str> = retry_reclaim_race(
            &RECLAIM_RETRY,
            true,
            || {
                calls.set(calls.get() + 1);
                Err(if calls.get() == 1 { OOM } else { "no device" })
            },
            |e| *e == OOM,
            || Some(SPARK_FULL_CACHE),
            |d| clock.set(clock.get() + d),
            || clock.get(),
            |_| {},
        );
        assert_eq!((r, calls.get()), (Err("no device"), 2));
        // And an error that is not out of memory is never retried, unified or not.
        let calls = Cell::new(0u32);
        let r: Result<(), &str> = retry_reclaim_race(
            &RECLAIM_RETRY,
            true,
            || {
                calls.set(calls.get() + 1);
                Err("no device")
            },
            |e| *e == OOM,
            || Some(SPARK_FULL_CACHE),
            |d| clock.set(clock.get() + d),
            || clock.get(),
            |_| {},
        );
        assert_eq!((r, calls.get()), (Err("no device"), 1));
    }

    /// `create_with`, the wiring `create_context` runs, over a scripted driver: the attribute
    /// read's answer, the creator's answers in order (the last one repeats), and the host.
    /// Returns the result's driver code (`Ok` is the number of the call that succeeded) and the
    /// calls made. Driver errors are compared by code and never formatted: cudarc asks the
    /// driver for their text, and a CPU test has none.
    fn wired(
        integrated: Result<i32, DriverError>,
        answers: &[Option<sys::CUresult>],
        host: Option<HostMem>,
    ) -> (Result<u32, sys::CUresult>, u32) {
        let calls = Cell::new(0u32);
        let clock = Cell::new(Duration::ZERO);
        let result = create_with(
            integrated,
            || {
                let i = calls.get() as usize;
                calls.set(calls.get() + 1);
                match answers[i.min(answers.len() - 1)] {
                    None => Ok(calls.get()),
                    Some(code) => Err(DriverError(code)),
                }
            },
            || host,
            |d| clock.set(clock.get() + d),
            || clock.get(),
            |_| {},
        );
        (result.map_err(|e| e.0), calls.get())
    }

    #[test]
    fn the_wiring_retries_only_a_unified_device_only_out_of_memory_only_above_the_floor() {
        use sys::cudaError_enum as E;
        let oom = Some(E::CUDA_ERROR_OUT_OF_MEMORY);
        let unified: Result<i32, DriverError> = Ok(1);
        // The GB10 shape: out of memory twice, then a context.
        assert_eq!(
            wired(unified, &[oom, oom, None], Some(SPARK_FULL_CACHE)),
            (Ok(3), 3)
        );
        // A discrete card (the attribute reads 0) keeps its first answer.
        assert_eq!(
            wired(Ok(0), &[oom, None], Some(SPARK_FULL_CACHE)),
            (Err(E::CUDA_ERROR_OUT_OF_MEMORY), 1)
        );
        // An attribute the driver would not read is a discrete card, not a unified one.
        assert_eq!(
            wired(
                Err(DriverError(E::CUDA_ERROR_INVALID_VALUE)),
                &[oom, None],
                Some(SPARK_FULL_CACHE)
            ),
            (Err(E::CUDA_ERROR_OUT_OF_MEMORY), 1)
        );
        // Another driver error is not retried on a unified device either.
        assert_eq!(
            wired(
                unified,
                &[Some(E::CUDA_ERROR_NO_DEVICE), None],
                Some(SPARK_FULL_CACHE)
            ),
            (Err(E::CUDA_ERROR_NO_DEVICE), 1)
        );
        // The serving policy's floor is the one applied: 100 MiB available is a short host.
        let short = HostMem {
            available: 100 << 20,
            ..SPARK_FULL_CACHE
        };
        assert_eq!(
            wired(unified, &[oom, None], Some(short)),
            (Err(E::CUDA_ERROR_OUT_OF_MEMORY), 1)
        );
        // And its budget: with a driver that never has the block, the tries end at 90 s.
        let (r, calls) = wired(unified, &[oom], Some(SPARK_FULL_CACHE));
        assert_eq!((r, calls), (Err(E::CUDA_ERROR_OUT_OF_MEMORY), 116));
    }

    #[test]
    fn the_lines_of_a_boot_that_recovers_carry_no_phrase_the_log_gates_scan_for() {
        // tools/serve-stress-gate.sh, tools/drafter-admission-gate.py and
        // tools/same-pass-admission-gate.py fail a serving cell on these anywhere in the server
        // log. A boot the retry saved is not a failure, so its lines must not trip them; a boot
        // that gives up or is not retried is one, and its lines must.
        const SCANNED: [&str; 7] = [
            "out of memory",
            "CUDA_ERROR",
            "panicked",
            "SIGSEGV",
            "memory allocation",
            "[worker] FATAL",
            "[worker] respawn",
        ];
        let retrying = retry_line(
            0,
            &RetryNote::Retrying {
                failed_tries: 1,
                elapsed: MS(180),
                wait: MS(50),
                host: SPARK_FULL_CACHE,
            },
        );
        let recovered = retry_line(
            0,
            &RetryNote::Recovered {
                tries: 3,
                elapsed: MS(750),
            },
        );
        for line in [&retrying, &recovered] {
            for phrase in SCANNED {
                assert!(!line.contains(phrase), "{line:?} contains {phrase:?}");
            }
            assert!(line.starts_with("[ctx-retry] device 0: "));
        }
        assert_eq!(
            retrying,
            "[ctx-retry] device 0: context creation returned driver code 2 (no memory) on unified \
             memory (try 1, 180 ms since the first try began; host MemFree 900 MiB, MemAvailable \
             121856 MiB, Cached 120832 MiB, free blocks of order 9 and up (2 MiB with 4 KiB \
             pages): 0); trying again in 50 ms"
        );
        assert_eq!(
            recovered,
            "[ctx-retry] device 0: context created on try 3, after 750 ms"
        );
        let gave_up = retry_line(
            0,
            &RetryNote::GaveUp {
                tries: 25,
                elapsed: MS(91_750),
                host: Some(SPARK_FULL_CACHE),
            },
        );
        assert!(gave_up.contains("still out of memory after 25 tries and 91750 ms"));
        assert!(gave_up.ends_with("; giving up"));
        for host in [Some(SPARK_FULL_CACHE), None] {
            let line = retry_line(0, &RetryNote::NotReclaimable { host });
            assert!(line.contains("out of memory") && line.ends_with("; not retried"));
        }
    }

    #[test]
    fn host_memory_is_parsed_from_its_own_lines() {
        let meminfo = "MemTotal:       127598768 kB\nMemFree:          934912 kB\n\
                       MemAvailable:   121878552 kB\nBuffers:            6144 kB\n\
                       Cached:         120832000 kB\nSwapCached:            0 kB\n";
        let buddy = "Node 0, zone      DMA     14     17     23     21     16     26     30     17     11     19     15      6      5      7 \n\
                     Node 0, zone   Normal   2305   4915   7239   5342   2958   2516    374     64     73      3      1      0      2      0 \n";
        let h = parse_host_mem(meminfo, Some(buddy)).expect("parsed");
        assert_eq!(h.free, 934_912 * 1024);
        assert_eq!(h.available, 121_878_552 * 1024);
        assert_eq!(h.cached, 120_832_000 * 1024, "Cached, not SwapCached");
        assert_eq!(
            h.large_free_blocks,
            Some(6),
            "orders 9 and up of the Normal zone only: the DMA zone's 52 are not counted"
        );
        assert_eq!(parse_host_mem("MemFree: 12 kB\n", None), None);
        assert_eq!(
            parse_host_mem(meminfo, None).unwrap().large_free_blocks,
            None
        );
        let text = h.to_string();
        assert!(text.contains("MemFree 913 MiB") && text.contains("MemAvailable 119022 MiB"));
        assert!(text.ends_with("free blocks of order 9 and up (2 MiB with 4 KiB pages): 6"));
    }

    #[test]
    fn a_failed_step_names_its_driver_call() {
        let e = boot_step::<(), _>(
            0,
            "cuStreamCreate (the compute stream)",
            Err(std::io::Error::other("boom")),
        )
        .unwrap_err();
        assert_eq!(
            e.to_string(),
            "cuStreamCreate (the compute stream) on device 0 failed: boom"
        );
    }

    #[test]
    fn only_a_driver_out_of_memory_answer_is_one() {
        // The retry and the host lines both hang on this test of the driver's code. (The error's
        // text is not formatted here: cudarc asks the driver for it, and a CPU test has none.)
        use sys::cudaError_enum as E;
        assert!(is_driver_oom(&DriverError(E::CUDA_ERROR_OUT_OF_MEMORY)));
        assert!(!is_driver_oom(&DriverError(E::CUDA_ERROR_NO_DEVICE)));
        assert!(!is_driver_oom(&DriverError(E::CUDA_ERROR_INVALID_VALUE)));
        // `boot_step` finds the driver error behind the generic bound and keeps its call name.
        let e = boot_step::<(), _>(3, "cuDeviceGet", Err(DriverError(E::CUDA_ERROR_NO_DEVICE)))
            .unwrap_err();
        let step = e.downcast_ref::<BootStepError>().expect("a named step");
        assert_eq!((step.ordinal, step.call), (3, "cuDeviceGet"));
        assert_eq!(
            step.host, None,
            "host lines are for out-of-memory answers only"
        );
        assert!(
            step.source
                .downcast_ref::<DriverError>()
                .is_some_and(|d| d.0 == E::CUDA_ERROR_NO_DEVICE)
        );
    }
}
