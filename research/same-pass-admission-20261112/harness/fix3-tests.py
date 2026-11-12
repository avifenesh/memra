#!/usr/bin/env python3
"""memra#1255 review round 1: the worker tests for S1, S2 (hold), nit 2 and S3. usage: fix3-tests.py WORKER_RS"""
import sys

p = sys.argv[1]
s = open(p).read()


def rep(old: str, new: str) -> None:
    global s
    n = s.count(old)
    if n != 1:
        sys.exit(f"needle found {n} times:\n{old[:200]}")
    s = s.replace(old, new)


rep("""        assert!(!fits);
        assert!(settle_wait(booked, fits));
        // Nothing owed: the gate's own verdicts decide, as before. A fit never waits.
        assert!(!settle_wait(0, false));
        assert!(!settle_wait(0, true));
        assert!(!settle_wait(booked, true));
    }""", """        assert!(!fits);
        assert!(settle_wait(booked, fits, true));
        // Nothing owed: the gate's own verdicts decide, as before. A fit never waits.
        assert!(!settle_wait(0, false, true));
        assert!(!settle_wait(0, true, true));
        assert!(!settle_wait(booked, true, true));
        // An arrival the reading does not hold even without the booking gains nothing from a
        // wait (the tick only lowers the reading): it goes on to the verdicts at once.
        assert!(!settle_wait(booked, false, false));
    }

    #[test]
    fn the_armed_door_books_the_whole_prompt_stack_of_an_unbuilt_session() {
        // memra#1255 review: on the armed memory door the unbuilt booking leaves each session's
        // prefill workspace to the pending-prime term, so that term must count the whole-prompt
        // stack of a session that owns no route state yet (a DFlash, gemma-spec or GLM cold
        // prime builds its walker in the tick). Two cold DFlash sessions of 5,741 and 5,772
        // rows: together the two terms book every charge less the workspace, each session's
        // stack, and one call (the slab and its returned rows: the calls run one after another).
        use super::{PrimeOwed, pending_prime_for_model, prime_walker_unstarted, unbuilt_charge_bytes};
        let shape = memra_engine::hybrid_forward::PrimeWorkspaceShape {
            call_row_bytes: 400_000,
            slab_row_bytes: 300_000,
            prompt_row_bytes: 20_480,
            n_layers: 64,
            swa_ring: false,
        };
        // Who owes the stack: an unbuilt session, an MTP session before its walker starts.
        assert!(prime_walker_unstarted(false, false, false), "unbuilt DFlash/GLM/gemma");
        assert!(prime_walker_unstarted(true, false, true), "MTP, walker not started");
        assert!(!prime_walker_unstarted(true, true, true), "MTP, walker started");
        assert!(!prime_walker_unstarted(false, false, true), "plain, cache built in admit");
        let rows = [5_741usize, 5_772];
        let call = |r: usize| shape.call_row_bytes * shape.call_rows(r);
        let stack = |r: usize| shape.prompt_row_bytes * r;
        let workspace = |r: usize| call(r) + stack(r);
        let kv_and_plane = 345_000_000usize;
        let sessions = rows.map(|r| (false, kv_and_plane + workspace(r), workspace(r)));
        let owed: Vec<PrimeOwed> = rows
            .iter()
            .map(|&r| PrimeOwed {
                rows: r,
                unstarted_walker: prime_walker_unstarted(false, false, false),
                cooperative: true,
                snapshot_bytes: 0,
            })
            .collect();
        let booked = unbuilt_charge_bytes(sessions.into_iter(), true)
            + pending_prime_for_model(&owed, Some(&shape), None, 0);
        let largest_call = rows.iter().map(|&r| call(r)).max().unwrap();
        let returned = shape.prompt_row_bytes * rows.iter().map(|&r| shape.call_rows(r)).max().unwrap();
        assert_eq!(
            booked,
            2 * kv_and_plane + stack(rows[0]) + stack(rows[1]) + largest_call + returned
        );
    }""")

rep("""        let settle = at("settle_wait(unbuilt_booked, headroom.sufficient(required))");""",
    """        let unbooked = at("pending_booked.saturating_sub(unbuilt_booked),");
        let settle =
            at("settle_wait(unbuilt_booked, headroom.sufficient(required), fits_unbooked)");""")
rep("""        assert!(booked < book && book < settle && settle < reclaim && reclaim < verdict);
        // The settle branch holds the line for later requests on its devices, as a VRAM defer
        // does, before it requeues (and before the reclaim, so nothing is decided past it).
        let hold = settle
            + prod[settle..]
                .find("memory_hold.extend(")
                .expect("settle hold");
        let requeued = settle + prod[settle..].find("requeue.push_back(req);").unwrap();
        assert!(hold < requeued && requeued < reclaim);
        assert!(verdict < recorded && recorded < pushed);""",
    """        assert!(booked < book && book < unbooked && unbooked < settle);
        assert!(settle < reclaim && reclaim < verdict);
        // The settle branch holds the line for later requests on its devices, as a VRAM defer
        // does, before it requeues (and before the reclaim, so nothing is decided past it).
        let hold = settle
            + prod[settle..]
                .find("memory_hold.extend(")
                .expect("settle hold");
        let requeued = settle + prod[settle..].find("requeue.push_back(req);").unwrap();
        let next = settle + prod[settle..].find("continue;").unwrap();
        assert!(hold < requeued && requeued < next && next < reclaim);
        // The hold lasts one pass: the set is made anew at the top of every admission pass.
        let pass = at("settled.retain(|_, (since, _, _)| since.elapsed() < Duration::from_secs(600));");
        let hold_set = at("let mut memory_hold: std::collections::HashSet<usize> = Default::default();");
        let drain = at("while let Some(mut req) = queue.pop_front() {");
        assert!(pass < hold_set && hold_set < drain && drain < settle);
        // The settle time ends when the request gets past the check, before the reclaim and
        // any VRAM defer; the `settled:` line prints only for a session `admit` built.
        let past = at("if let Some((since, passes)) = settling.remove(&req.request_id) {");
        assert!(next < past && past < reclaim);
        let line = at("\\"[admission] settled: model={:?} admitted after {passes} settle \\\\");
        assert!(verdict < recorded && recorded < line && line < pushed);""")
open(p, "w").write(s)
print("tests patched")
