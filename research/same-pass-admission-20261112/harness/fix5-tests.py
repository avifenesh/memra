#!/usr/bin/env python3
"""memra#1255 review round 2: tests. usage: fix5-tests.py WORKER_RS"""
import sys

p = sys.argv[1]
s = open(p).read()


def rep(old: str, new: str) -> None:
    global s
    n = s.count(old)
    if n != 1:
        sys.exit(f"needle found {n} times:\n{old[:240]}")
    s = s.replace(old, new)


# my source-order test: the unbooked reading's call and the Interactive guard on the hold
rep('''        let unbooked = at("pending_booked.saturating_sub(unbuilt_booked),");''',
    '''        let unbooked = at("unbooked_pending(\\n                            pending_booked,\\n                            unbuilt_booked,\\n                            pending_prime,\\n                            pending_prime_built,\\n                        ),");''')
rep('''        let next = settle + prod[settle..].find("continue;").unwrap();
        assert!(hold < requeued && requeued < next && next < reclaim);''',
    '''        let next = settle + prod[settle..].find("continue;").unwrap();
        assert!(hold < requeued && requeued < next && next < reclaim);
        // Only an interactive settle wait holds the line, on its own model's devices.
        let guarded = at("if lane == crate::lanes::Lane::Interactive {\\n                        memory_hold.extend(model_devices(&engine, &loaded[&model_key]));");
        assert!(settle < guarded && guarded < hold && hold < requeued);''')

# the reclaim booking's wiring test
rep('''        let prime = at(
            "let prime_owed = pending_prime_owed_bytes(&engine, &loaded, &active, &admission_costs);",
        );
        let disarm = at(
            "if reclaim_booking_disarms(reclaim_booking, prime_owed, pending_seed_uncapped) { \\
             reclaim_booking = false;",
        );''',
    '''        let prime = at(
            "let prime_owed_any = pending_prime_owed_bytes(&engine, &loaded, &active, \\
             &admission_costs, false);",
        );
        // memra#1255: the booked amount leaves out unbuilt sessions where the unbuilt booking
        // applies; whether anything is owed still counts them.
        let skipped = at(
            "let prime_owed = if unbuilt_applies { pending_prime_owed_bytes(&engine, &loaded, \\
             &active, &admission_costs, true) } else { prime_owed_any };",
        );
        let disarm = at(
            "if reclaim_booking_disarms(reclaim_booking, prime_owed_any, pending_seed_uncapped) { \\
             reclaim_booking = false;",
        );''')
rep('''        assert!(booked < prime && prime < disarm && disarm < booking);''',
    '''        assert!(booked < prime && prime < skipped && skipped < disarm && disarm < booking);''')
rep('''        let owed = at(
            "let owed = reclaim_owed_bytes(&engine, &loaded, &active, &admission_costs) \\
             .saturating_add(reclaim_arrival_seed_bytes( prompt_len, \\
             model_prefix_entry_bytes(model, prompt_len), ));",
        );''',
    '''        let owed = at(
            "let owed = reclaim_owed_bytes( &engine, &loaded, &active, &admission_costs, \\
             unbuilt_applies, ) .saturating_add(reclaim_arrival_seed_bytes( prompt_len, \\
             model_prefix_entry_bytes(model, prompt_len), ));",
        );''')
rep('''            "pending_prime_owed_bytes(engine,loaded,active,admission_costs).saturating_add(pending_seed_bytes(active))"''',
    '''            "pending_prime_owed_bytes(engine,loaded,active,admission_costs,skip_unbuilt).saturating_add(pending_seed_bytes(active))"''')

# the new CPU test, after the armed-door stack test
rep('''    #[test]
    fn every_route_state_field_counts_as_built_and_the_settle_wait_precedes_every_verdict() {''',
    '''    #[test]
    fn an_unbuilt_session_is_booked_once_and_left_out_of_the_reading_without_the_booking() {
        // memra#1255 review round 2. One cold DFlash session S that owns no route state yet and
        // one plain session P still priming. With the door off after a partial reclaim, the
        // reclaim booking counts what P owes and not S (S's whole charge is the unbuilt
        // booking); the reading without the unbuilt booking is the raw reading less P's term
        // alone; the partial reclaim's need counts S's workspace once. On the armed door, the
        // unbooked reading also drops S's share of the pending-prime term.
        use super::{
            PrimeOwed, pending_prime_for_model, prime_owed_counts, prime_walker_unstarted,
            reclaim_owed_booking, unbooked_pending, unbuilt_charge_bytes,
        };
        let shape = memra_engine::hybrid_forward::PrimeWorkspaceShape {
            call_row_bytes: 400_000,
            slab_row_bytes: 300_000,
            prompt_row_bytes: 20_480,
            n_layers: 64,
            swa_ring: false,
        };
        let owed_of = |rows: usize, owns: bool, mtp: bool| PrimeOwed {
            rows,
            unstarted_walker: prime_walker_unstarted(mtp, false, owns),
            cooperative: true,
            snapshot_bytes: 0,
        };
        // (owns_route_state, PrimeOwed): S unbuilt (5,741 rows), P a plain session priming.
        let sessions = [(false, owed_of(5_741, false, false)), (true, owed_of(1_300, true, false))];
        let v2 = |skip_unbuilt: bool| {
            let owed: Vec<PrimeOwed> = sessions
                .iter()
                .filter(|(owns, _)| prime_owed_counts(skip_unbuilt, *owns))
                .map(|(_, o)| *o)
                .collect();
            pending_prime_for_model(&owed, Some(&shape), None, 0)
        };
        let p_only = pending_prime_for_model(&[sessions[1].1], Some(&shape), None, 0);
        let s_workspace = shape.call_row_bytes * shape.call_rows(5_741) + shape.prompt_row_bytes * 5_741;
        let s_charge = 345_000_000 + s_workspace;
        let budget = 1 << 40;

        // Door off, single device, the reclaim booking armed by a partial reclaim.
        assert_eq!(v2(true), p_only, "the booked owed term leaves S out");
        let reclaim_booked = reclaim_owed_booking(true, v2(true), 0, budget, 0);
        let unbuilt = unbuilt_charge_bytes([(false, s_charge, s_workspace)].into_iter(), false);
        assert_eq!(unbuilt, s_charge);
        let pending_booked = reclaim_booked + unbuilt;
        // The reading without the booking: the raw reading less the other bookings only.
        assert_eq!(unbooked_pending(pending_booked, unbuilt, 0, 0), p_only);
        // The partial reclaim's need: the bookings less the reclaim booking, plus what is owed,
        // counts S's charge (its workspace included) once.
        let need_bookings = pending_booked - reclaim_booked + v2(true);
        assert_eq!(need_bookings, s_charge + p_only);
        // Counting S in the owed term too would add its stack and call a second time.
        assert!(pending_booked - reclaim_booked + v2(false) > s_charge + p_only + 5_741 * 20_480);

        // Armed door: the booked reading keeps S's stack in the pending-prime term and leaves
        // its workspace out of the unbuilt booking; the reading without the booking drops both.
        let armed_unbuilt = unbuilt_charge_bytes([(false, s_charge, s_workspace)].into_iter(), true);
        let pending_prime = v2(false);
        let pending_prime_built = v2(true);
        let armed_booked = pending_prime + armed_unbuilt;
        assert_eq!(
            unbooked_pending(armed_booked, armed_unbuilt, pending_prime, pending_prime_built),
            p_only
        );
    }

    #[test]
    fn every_route_state_field_counts_as_built_and_the_settle_wait_precedes_every_verdict() {''')
open(p, "w").write(s)
print("tests patched")
