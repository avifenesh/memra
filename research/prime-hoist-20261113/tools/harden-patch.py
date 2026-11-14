#!/usr/bin/env python3
"""memra#1258 step 2, the park floor: a prime's GDN scan workspace is parked at the prime's end
only while the device's free reading holds the floor admission keeps free; under it the
workspace is dropped, so its bytes are back in the pool for the steps after the prime.

Applies the same source edits to a tree's engine lib.rs and server worker.rs (the branch's, and
the diagnostic capacity arm's on its older base). With --tests it also adds the tests (the
branch only: the diagnostic arm is a server binary).
usage: harden-patch.py [--tests] LIB_RS WORKER_RS [HYBRID_FORWARD_RS]"""
import sys

args = [a for a in sys.argv[1:] if a != "--tests"]
tests = "--tests" in sys.argv[1:]
lib_p, worker_p = args[0], args[1]
hf_p = args[2] if len(args) > 2 else None


def sub(s: str, old: str, new: str) -> str:
    assert s.count(old) == 1, (s.count(old), old[:90])
    return s.replace(old, new)


lib = open(lib_p).read()
assert "PRIME_GDN_WS_PARK_FLOOR" not in lib, "already patched"

# 1. The floor, its setter and the decision, after GdnScanWs's impl.
lib = sub(
    lib,
    "\n#[cfg(test)]\nmod gdn_scan_ws_tests {\n",
    '''
/// The free-memory floor under which a prime's GDN scan workspace is not parked (memra#1258).
/// 0, the default and what an engine binary without admission runs with, parks always.
static PRIME_GDN_WS_PARK_FLOOR: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Sets the floor a prime's GDN scan workspace needs free on its device to stay parked between
/// primes. A server passes the floor its admission keeps free in front of every request
/// (`MEMRA_ADMIT_DRIVER_HEADROOM_MB`): parked bytes count as cached, but only a prime or a trim
/// can use them, so on a device under that floor they go back to the pool instead.
pub fn set_prime_gdn_ws_park_floor(bytes: usize) {
    PRIME_GDN_WS_PARK_FLOOR.store(bytes, std::sync::atomic::Ordering::Relaxed);
}

/// Whether a prime's GDN scan workspace stays parked at the prime's end: always with no floor;
/// with one, only while the device's free reading (`None`: unreadable) holds it.
pub(crate) fn prime_gdn_ws_parks(free: Option<usize>, floor: usize) -> bool {
    floor == 0 || free.is_some_and(|free| free >= floor)
}

#[cfg(test)]
mod gdn_scan_ws_tests {
''',
)

# 2. The park.
lib = sub(
    lib,
    '''    /// Parks a prime's GDN scan workspace between primes (memra#1258): it stays mapped for the
    /// next prime and counts as reclaimable until a trim frees it.
    pub(crate) fn prime_gdn_ws_park(&self, ws: Option<GdnScanWs>) {
        if ws.is_some() {
            *self
                .prime_gdn_ws_parked
                .lock()
                .unwrap_or_else(|p| p.into_inner()) = ws;
        }
    }
''',
    '''    /// Parks a prime's GDN scan workspace between primes (memra#1258): it stays mapped for the
    /// next prime and counts as reclaimable until a trim frees it.
    ///
    /// Only a prime or a trim can use parked bytes. The steps that follow a prime allocate from
    /// the pool's cached blocks and, past those, from the driver. So the workspace is parked only
    /// while the device's free reading (admission's, `allocatable_mem_info`) holds the park floor
    /// (`set_prime_gdn_ws_park_floor`). Under it the workspace is dropped here and its bytes are
    /// cached blocks again, as the per-layer planes were before the workspace: the next prime
    /// allocates a new one.
    pub(crate) fn prime_gdn_ws_park(&self, ws: Option<GdnScanWs>) {
        let Some(ws) = ws else {
            return;
        };
        let floor = PRIME_GDN_WS_PARK_FLOOR.load(std::sync::atomic::Ordering::Relaxed);
        let free = (floor > 0)
            .then(|| crate::unified_memory::allocatable_mem_info(self.ctx()).ok())
            .flatten()
            .map(|(free, _)| free);
        if !prime_gdn_ws_parks(free, floor) {
            eprintln!(
                "[prime-ws] dev{} GDN scan workspace not parked: free_bytes={} floor_bytes={floor} \\
                 workspace_bytes={}",
                self.ctx().ordinal(),
                free.map_or_else(|| "unreadable".to_string(), |free| free.to_string()),
                ws.bytes(),
            );
            return;
        }
        *self
            .prime_gdn_ws_parked
            .lock()
            .unwrap_or_else(|p| p.into_inner()) = Some(ws);
    }
''',
)

if tests:
    # 3. The decision's unit test.
    lib = sub(
        lib,
        '''        // Qwen3.8-27B (48 value heads, kb16 sized for 48): 123,072 B/row, 504 MB at 4,096 rows.
        assert_eq!(GdnScanWs::row_bytes(48, 48, true), 123_072);
    }
}
''',
        '''        // Qwen3.8-27B (48 value heads, kb16 sized for 48): 123,072 B/row, 504 MB at 4,096 rows.
        assert_eq!(GdnScanWs::row_bytes(48, 48, true), 123_072);
    }

    /// memra#1258: the workspace stays parked only while the device holds the park floor. With
    /// no floor it always parks; under the floor, or on a device that cannot be read, it does not
    /// (94 MB is the driver's free reading after a prime in the margin sweep's zero-margin row).
    #[test]
    fn the_workspace_parks_only_above_the_floor() {
        let floor = 2048usize << 20;
        assert!(super::prime_gdn_ws_parks(None, 0));
        assert!(super::prime_gdn_ws_parks(Some(0), 0));
        assert!(super::prime_gdn_ws_parks(Some(floor), floor));
        assert!(super::prime_gdn_ws_parks(Some(floor + 1), floor));
        assert!(!super::prime_gdn_ws_parks(Some(floor - 1), floor));
        assert!(!super::prime_gdn_ws_parks(Some(94 << 20), floor));
        assert!(!super::prime_gdn_ws_parks(None, floor));
    }
}
''',
    )
open(lib_p, "w").write(lib)

worker = open(worker_p).read()
assert "set_prime_gdn_ws_park_floor" not in worker, "already patched"
# 4. The server hands the engine admission's floor at boot, before any prime.
worker = sub(
    worker,
    "    // PREFILL WORKSPACE HELD (research/c8-workspace-20261108): the card's arm, resolved once.\n",
    '''    // PARKED PRIME WORKSPACE (memra#1258): a prime parks its GDN scan workspace only on a device
    // that holds the floor admission keeps free; under it the bytes go back to the pool for the
    // steps that follow the prime.
    memra_engine::set_prime_gdn_ws_park_floor(admit_driver_headroom_bytes());
    // PREFILL WORKSPACE HELD (research/c8-workspace-20261108): the card's arm, resolved once.
''',
)
if tests:
    worker = sub(
        worker,
        '''    #[test]
    fn reserve_override_door_binds_both_paths() {
''',
        '''    #[test]
    fn the_worker_hands_the_engine_the_park_floor() {
        // memra#1258: admission counts a parked prime workspace as cached, and only a prime or
        // a trim can use it. The engine parks it only above the floor the driver-headroom rung
        // keeps; without this call the floor is 0 and it parks on a short device too.
        let src = include_str!("worker.rs");
        let flat: String = src.chars().filter(|c| !c.is_whitespace()).collect();
        assert!(flat.contains(concat!(
            "memra_engine::set_prime_gdn_ws_park",
            "_floor(admit_driver_headroom_bytes());"
        )));
    }

    #[test]
    fn reserve_override_door_binds_both_paths() {
''',
    )
open(worker_p, "w").write(worker)

if hf_p and tests:
    hf = open(hf_p).read()
    # 5. The census: the park consults the decision on admission's reading.
    hf = sub(
        hf,
        '''        let second = trim.rfind(call).expect("a second trim");
        assert!(first < check && check < parked && parked < second, "{trim}");
''',
        '''        let second = trim.rfind(call).expect("a second trim");
        assert!(first < check && check < parked && parked < second, "{trim}");
        // The park keeps the workspace only above the park floor, on admission's free reading.
        let park = code(fn_body(
            &lib,
            concat!("pub(crate) fn prime_gdn_ws", "_park("),
        ));
        assert!(
            park.contains(concat!("if !prime_gdn_ws", "_parks(free, floor)"))
                && park.contains(concat!("allocatable_mem", "_info(self.ctx())")),
            "{park}"
        );
''',
    )
    open(hf_p, "w").write(hf)
print("patched", lib_p, worker_p, hf_p or "")
