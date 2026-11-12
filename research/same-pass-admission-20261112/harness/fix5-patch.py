#!/usr/bin/env python3
"""memra#1255 review round 2 (worker.rs). Should-fix 1: the reading without the unbuilt booking
is built from terms over sessions that own route state only, and with the door off on a
single-device deployment the reclaim booking's and the partial reclaim's owed terms leave out
unbuilt sessions (their whole charge is in the unbuilt booking). Nit 2: the head-of-line docs name
settle waits. Nit 3: the test pins the hold's Interactive guard. usage: fix5-patch.py WORKER_RS"""
import sys

p = sys.argv[1]
s = open(p).read()


def rep(old: str, new: str, count: int = 1) -> None:
    global s
    n = s.count(old)
    if n != count:
        sys.exit(f"needle found {n} times, wanted {count}:\n{old[:240]}")
    s = s.replace(old, new)


# --- pure parts ---
rep("""/// One still-priming session as the corrected pending-prime term reads it (WP-B day 39).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PrimeOwed {""",
    """/// Whether the pending-prime term counts a session (memra#1255). `skip_unbuilt` leaves out a
/// session that owns no route state yet: with the memory door off on a single-device deployment
/// its whole charge, workspace included, is in the unbuilt booking already, and the reading
/// without the unbuilt booking must not carry any of it.
fn prime_owed_counts(skip_unbuilt: bool, owns_route_state: bool) -> bool {
    !skip_unbuilt || owns_route_state
}

/// The bookings the reading without the unbuilt booking subtracts (memra#1255): every booking
/// less the unbuilt charges and less the unbuilt sessions' share of the pending-prime term (the
/// term over every session less the term over the sessions that own route state).
fn unbooked_pending(
    pending_booked: usize,
    unbuilt_booked: usize,
    pending_prime: usize,
    pending_prime_built: usize,
) -> usize {
    pending_booked
        .saturating_sub(unbuilt_booked)
        .saturating_sub(pending_prime.saturating_sub(pending_prime_built))
}

/// One still-priming session as the corrected pending-prime term reads it (WP-B day 39).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PrimeOwed {""")

rep("""fn pending_prime_bytes_v2(
    active: &[Session],
    admission_costs: &HashMap<String, AdmissionCostModel>,
    slab_bytes: &dyn Fn(&str) -> usize,
    cooperative: &dyn Fn(&str) -> bool,
) -> usize {
    let mut by_model: HashMap<&str, Vec<PrimeOwed>> = HashMap::new();
    for s in active {
        let rows = session_pending_prime_rows(s.prefill_done, s.prefill_queue.len());""",
    """fn pending_prime_bytes_v2(
    active: &[Session],
    admission_costs: &HashMap<String, AdmissionCostModel>,
    slab_bytes: &dyn Fn(&str) -> usize,
    cooperative: &dyn Fn(&str) -> bool,
    skip_unbuilt: bool,
) -> usize {
    let mut by_model: HashMap<&str, Vec<PrimeOwed>> = HashMap::new();
    for s in active {
        if !prime_owed_counts(skip_unbuilt, session_owns_route_state(s)) {
            continue;
        }
        let rows = session_pending_prime_rows(s.prefill_done, s.prefill_queue.len());""")

rep("""/// The prime transients the sessions still priming will allocate after this reading
/// (`pending_prime_bytes_v2`: one call's slab growth and returned rows, the unstarted walkers'
/// stacks, the checkpoints the primes will take), with the gate's own slab and walker readings.
fn pending_prime_owed_bytes(
    engine: &Engine,
    loaded: &HashMap<String, LoadedModel>,
    active: &[Session],
    admission_costs: &HashMap<String, AdmissionCostModel>,
) -> usize {""",
    """/// The prime transients the sessions still priming will allocate after this reading
/// (`pending_prime_bytes_v2`: one call's slab growth and returned rows, the unstarted walkers'
/// stacks, the checkpoints the primes will take), with the gate's own slab and walker readings.
/// `skip_unbuilt` leaves out the sessions that own no route state yet (`prime_owed_counts`).
fn pending_prime_owed_bytes(
    engine: &Engine,
    loaded: &HashMap<String, LoadedModel>,
    active: &[Session],
    admission_costs: &HashMap<String, AdmissionCostModel>,
    skip_unbuilt: bool,
) -> usize {""")
rep("""                    .is_some_and(|lm| lm.model.mtp_prime_walk_supported())
        },
    )
}

/// What the sessions still priming will allocate after this reading, which the door-off gate does""",
    """                    .is_some_and(|lm| lm.model.mtp_prime_walk_supported())
        },
        skip_unbuilt,
    )
}

/// What the sessions still priming will allocate after this reading, which the door-off gate does""")
rep("""fn reclaim_owed_bytes(
    engine: &Engine,
    loaded: &HashMap<String, LoadedModel>,
    active: &[Session],
    admission_costs: &HashMap<String, AdmissionCostModel>,
) -> usize {
    pending_prime_owed_bytes(engine, loaded, active, admission_costs)
        .saturating_add(pending_seed_bytes(active))
}""",
    """fn reclaim_owed_bytes(
    engine: &Engine,
    loaded: &HashMap<String, LoadedModel>,
    active: &[Session],
    admission_costs: &HashMap<String, AdmissionCostModel>,
    skip_unbuilt: bool,
) -> usize {
    pending_prime_owed_bytes(engine, loaded, active, admission_costs, skip_unbuilt)
        .saturating_add(pending_seed_bytes(active))
}""")

# --- the admission pass ---
rep("""                let pending_prime_v1 = if admit_memory_cfg.armed {
                    pending_prime_bytes(&active, &admission_costs)
                } else {
                    0
                };
                let pending_prime = if admit_memory_cfg.armed {
                    pending_prime_bytes_v2(
                        &active,
                        &admission_costs,
                        &|name| {
                            loaded
                                .get(name)
                                .map_or(0, |lm| lm.model.prime_slab_bytes(&engine))
                        },
                        &|name| {
                            memra_engine::prime_walker::prime_yield_enabled()
                                && loaded
                                    .get(name)
                                    .is_some_and(|lm| lm.model.mtp_prime_walk_supported())
                        },
                    )
                } else {
                    0
                };""",
    """                let pending_prime_v1 = if admit_memory_cfg.armed {
                    pending_prime_bytes(&active, &admission_costs)
                } else {
                    0
                };
                // memra#1255: the unbuilt booking applies on a single-device deployment only.
                let unbuilt_applies = !is_multi_device_deployment(&loaded);
                let prime_slab = |name: &str| {
                    loaded
                        .get(name)
                        .map_or(0, |lm| lm.model.prime_slab_bytes(&engine))
                };
                let prime_coop = |name: &str| {
                    memra_engine::prime_walker::prime_yield_enabled()
                        && loaded
                            .get(name)
                            .is_some_and(|lm| lm.model.mtp_prime_walk_supported())
                };
                let pending_prime = if admit_memory_cfg.armed {
                    pending_prime_bytes_v2(
                        &active,
                        &admission_costs,
                        &prime_slab,
                        &prime_coop,
                        false,
                    )
                } else {
                    0
                };
                // The same term over the sessions that own route state only, for the reading
                // without the unbuilt booking (memra#1255, `unbooked_pending`).
                let pending_prime_built = if admit_memory_cfg.armed && unbuilt_applies {
                    pending_prime_bytes_v2(
                        &active,
                        &admission_costs,
                        &prime_slab,
                        &prime_coop,
                        true,
                    )
                } else {
                    pending_prime
                };""")
rep("""                let reclaim_booked = if reclaim_booking && !admit_memory_cfg.armed {
                    let prime_owed =
                        pending_prime_owed_bytes(&engine, &loaded, &active, &admission_costs);
                    if reclaim_booking_disarms(reclaim_booking, prime_owed, pending_seed_uncapped) {""",
    """                let reclaim_booked = if reclaim_booking && !admit_memory_cfg.armed {
                    // memra#1255: whether anything is still owed counts every priming session;
                    // what is booked leaves out the sessions with no route state yet, whose
                    // whole charge the unbuilt booking carries.
                    let prime_owed_any =
                        pending_prime_owed_bytes(&engine, &loaded, &active, &admission_costs, false);
                    let prime_owed = if unbuilt_applies {
                        pending_prime_owed_bytes(&engine, &loaded, &active, &admission_costs, true)
                    } else {
                        prime_owed_any
                    };
                    if reclaim_booking_disarms(reclaim_booking, prime_owed_any, pending_seed_uncapped)
                    {""")
rep("""                let unbuilt_booked = if is_multi_device_deployment(&loaded) {
                    0
                } else {
                    unbuilt_session_booking(&active, admit_memory_cfg.armed)
                };""",
    """                let unbuilt_booked = if unbuilt_applies {
                    unbuilt_session_booking(&active, admit_memory_cfg.armed)
                } else {
                    0
                };""")
rep("""                let fits_unbooked = raw_headroom.clone().is_some_and(|h| {
                    h.less_pending(
                        pending_booked.saturating_sub(unbuilt_booked),
                        primary_device,
                    )
                    .sufficient(required)
                });""",
    """                let fits_unbooked = raw_headroom.clone().is_some_and(|h| {
                    h.less_pending(
                        unbooked_pending(
                            pending_booked,
                            unbuilt_booked,
                            pending_prime,
                            pending_prime_built,
                        ),
                        primary_device,
                    )
                    .sufficient(required)
                });""")
rep("""                            let owed =
                                reclaim_owed_bytes(&engine, &loaded, &active, &admission_costs)
                                    .saturating_add(reclaim_arrival_seed_bytes(""",
    """                            // memra#1255: an unbuilt session's whole charge is in
                            // `pending_booked` (the unbuilt booking); the owed term leaves it out.
                            let owed = reclaim_owed_bytes(
                                &engine,
                                &loaded,
                                &active,
                                &admission_costs,
                                unbuilt_applies,
                            )
                            .saturating_add(reclaim_arrival_seed_bytes(""")

# --- nit 2: head-of-line docs ---
rep("""    /// admission pass (head-of-line memory order), counted per pass like the defers.
    pub admission_vram_hol_holds: u64,""",
    """    /// admission pass (head-of-line memory order), counted per pass like the defers. The hold
    /// sits behind a VRAM-deferred or a settle-waiting request (memra#1255).
    pub admission_vram_hol_holds: u64,""")

open(p, "w").write(s)
print("patched")
