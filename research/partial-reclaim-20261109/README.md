# A partial admission reclaim on a discrete card (2026-11-09)

Follow-up to the discrete prefix budget (`docs/decisions/PREFIX-CACHE-DISCRETE.md`, Owed: "a partial admission
reclaim at a large budget"). The acceptance test is that lane's at-the-limit cell
(`research/prefix-budget-20261107/pro6000-maxq/limit/`).

## Question

With the memory door off, a deferred arrival runs `reclaim-on-defer`, which drops every unleased prefix entry. The
discrete budget now leaves the session peak, so on one RTX PRO 6000 Max-Q at `MEMRA_CTX=32768` and
`MEMRA_MAX_SESSIONS=12` the cache may hold 54,277,046,272 B. In the at-the-limit cell (`limit-2-B`, `limit-3-B`) the
seventh of twelve pressure arrivals deferred with six admitted sessions still priming. Admission charged it 3,435 MB
(KV at 25,617 tokens, a 2,317 MB prefill workspace and 357 MB fixed) plus the transient reserve; the shortfall was
under 1 GB. The reclaim dropped 57 entries (54.2 GB, effective free 4,606 MB to 58,802 MB), and the 16 repeats that
followed re-primed: 0 hits, 10.6 s each.

## Design

The reclaim site is the one in `crates/memra-server/src/worker.rs` that reads "Drop unleased snapshots before
deciding to queue or reject".

1. **Where it applies.** The memory door off, a discrete card, a single primary reading, and a derived budget above
   its two-entry floor. Everything else keeps the flush: the door's demoting path (armed only on the full-power
   RTX PRO 6000 Workstation), unified memory (GB10), a per-device reading (PP, multi-device plans) and an operator
   budget (`MEMRA_PREFIX_CACHE_MB`). The floor regime keeps it on purpose: at 2.26 GB the cache holds two entries,
   so a partial pass can keep at most one of them, and the lane before this one measured the partial pass flat
   there (RTX 5090, `phase1a-final/`: 2, 2, 3 repeats against 3, 3, 3). Every RTX 5090 configuration and the Max-Q
   at the default caps sit on the floor, so their behavior is byte-identical.
2. **The pass.** Free the arrival's shortfall plus what the sessions still priming will allocate, oldest entry
   first, and keep the rest. The shortfall is read unbooked (the booked reading floors at 0 and would understate it).
   The owed bytes are the prime transients (`pending_prime_bytes_v2`: one call's slab growth and returned rows, the
   unstarted walkers' stacks, the checkpoints the primes will take) and the prefix entries the priming sessions
   will publish (`pending_seed_bytes`). The seeds are counted uncapped: before the pass a full cache replaces bytes
   when a seed lands, but the pass makes room, and the seeds regrow into it. Eviction is `evict_to_bytes`, the
   cache's one victim order (plain LRU, `PREFIX-CACHE-POLICY.md`); it never takes a leased entry, so the arrival's
   donor and the restorable spare's pins stay. Then the settle step (memra#346) hands the bytes to the driver, and
   the headroom is read again.
3. **The fallback.** If the arrival is still short (allocator rounding, a neighbour's growth between readings, or
   leased bytes the pass could not reach), the rest of the unleased cache goes, as before. The restorable spare's
   climb and the parked-session ladder then run unchanged, on the empty evictable set they assume.
4. **The booking.** The flush hid a gap in the door-off gate: it books nothing for admitted sessions that are still
   priming. A partial pass leaves free only the arrival's need plus what the priming sessions owe, so the next
   arrival in the same burst passes the gate on the owed bytes (each arrival is charged a full prefill workspace,
   about 3.4 GB here, but the shared slab means it allocates about 1 GB at admission). The seeds then land into the
   room the pass made. In the limit cell's burst that is about 5.5 GB of seeds for the six sessions priming at the
   defer and 11 GB for all twelve, against a few GB of free memory. So once a pass has kept cache bytes, admission
   books the seeds the priming sessions will publish, capped at the room the cache has left (`seed_booking_cap`, the
   door's own seed term), until no session has a seed pending. A later arrival then defers and runs its own partial
   pass. The prime transients are not booked: the door-off gate already covers them through each arrival's own
   workspace charge, and booking them would defer arrivals on a full cache, where no seed adds bytes. While the cache
   is at its budget the booking is 0, whatever is pending.
5. **No flag.** The rollback is the revert.

The lane's banked code (`bank/prefix-budget-gds-20261107`, 39ec446dd5: `reclaim_prefix_target`,
`reclaim_owed_bytes`) had the pass and the owed term, on an older reclaim site with the off-tick and demote arms. It
had no booking and applied on every discrete card. This change re-derives the pass against the current site and
adds the floor key and the booking.

## Expected on the limit cell

The pressure burst defers once per arrival from the seventh on, and each pass frees that arrival's shortfall plus
the seeds then pending: roughly 15 to 20 GB over the burst, oldest fill entries first. The cache keeps the 16 most
recent fill prompts, and the repeats hit them. No fallback flush, no OOM line, pressure TTFT within noise of A (the
passes are synchronous and cheap beside 25k-token prefills).

## Acceptance (wrx, one RTX PRO 6000 Max-Q, untimed)

- Arms: A = #1090's head (the flush), B = this change. Interleaved A B B A, `harness/limit.py` through the lane's arm
  script, `MEMRA_CTX=32768`, `MEMRA_MAX_SESSIONS=12`.
- Pass: B's repeats hit most of the 16; no OOM line; pressure TTFT not worse than A beyond noise; the flush gone or
  rare (every reclaim line counted with what it freed).
- Regression: the RTX 5090 lane's mixed shape (`harness/mixed.py`, `MEMRA_MAX_SESSIONS=12`) on the same card class,
  A against B, three runs per arm, counts only.
- A negative result keeps the code on a research branch and is written up here.
