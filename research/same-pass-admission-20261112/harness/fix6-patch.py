#!/usr/bin/env python3
"""memra#1255 review round 3 (worker.rs): pin the skip's four production literals (the armed
door's two v2 calls with their booleans, the filter inside v2's body, and the owns-route-state
argument v2 passes to prime_walker_unstarted); the partial-reclaim line counts only the priming
sessions the owed term keeps. usage: fix6-patch.py WORKER_RS"""
import sys

p = sys.argv[1]
s = open(p).read()


def rep(old: str, new: str) -> None:
    global s
    n = s.count(old)
    if n != 1:
        sys.exit(f"needle found {n} times:\n{old[:240]}")
    s = s.replace(old, new)


# nit 1: the partial-reclaim line's session count
rep("""                                    owed as f64 / 1e6,
                                    active
                                        .iter()
                                        .filter(|s| {
                                            session_pending_prime_rows(
                                                s.prefill_done,
                                                s.prefill_queue.len(),
                                            ) > 0
                                        })
                                        .count(),""",
    """                                    owed as f64 / 1e6,
                                    // The priming sessions the owed term counts (memra#1255:
                                    // not the ones with no route state yet, whose charge the
                                    // unbuilt booking carries).
                                    active
                                        .iter()
                                        .filter(|s| {
                                            session_pending_prime_rows(
                                                s.prefill_done,
                                                s.prefill_queue.len(),
                                            ) > 0
                                                && prime_owed_counts(
                                                    unbuilt_applies,
                                                    session_owns_route_state(s),
                                                )
                                        })
                                        .count(),""")

# the pins, in the source-order test after the guard pin
rep("""        assert!(settle < guarded && guarded < hold && hold < requeued);""",
    """        assert!(settle < guarded && guarded < hold && hold < requeued);
        // The skip's literals (review round 3): the armed door's booked reading counts every
        // session in the pending-prime term (`false`), the reading without the booking only the
        // built ones (`true`); v2's body applies the skip and passes the predicate to
        // `prime_walker_unstarted`. Whitespace-squashed, booleans included.
        let squashed: String = prod.split_whitespace().collect::<Vec<_>>().join(" ");
        let pin = |needle: &str| {
            assert_eq!(squashed.matches(needle).count(), 1, "{needle}");
            squashed.find(needle).unwrap()
        };
        let armed = pin(
            "let pending_prime = if admit_memory_cfg.armed { pending_prime_bytes_v2( &active, \\
             &admission_costs, &prime_slab, &prime_coop, false, ) } else { 0 };",
        );
        let built = pin(
            "let pending_prime_built = if admit_memory_cfg.armed && unbuilt_applies { \\
             pending_prime_bytes_v2( &active, &admission_costs, &prime_slab, &prime_coop, true, ) \\
             } else { pending_prime };",
        );
        assert!(armed < built);
        let v2_body = pin("fn pending_prime_bytes_v2(");
        let filter = pin(
            "for s in active { if !prime_owed_counts(skip_unbuilt, session_owns_route_state(s)) { \\
             continue; }",
        );
        let walker = pin(
            "unstarted_walker: prime_walker_unstarted( s.spec.is_some(), s.mtp_prime.is_some(), \\
             session_owns_route_state(s), ),",
        );
        assert!(v2_body < filter && filter < walker);""")
open(p, "w").write(s)
print("patched")
