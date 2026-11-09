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
   budget (`MEMRA_PREFIX_CACHE_MB`). The floor keeps it on purpose: at 2.26 GB the cache holds two entries, so a
   partial pass can keep at most one of them, and the lane before this one measured the partial pass flat there
   (RTX 5090, `phase1a-final/`: 2, 2, 3 repeats against 3, 3, 3). The key is the derived budget, not the card. Every
   measured RTX 5090 configuration (Qwen3.8-27B NVFP4 + Q5_K at 32,768 tokens, `MEMRA_MAX_SESSIONS=12` or the
   default caps) and the Max-Q at the default caps sit on the floor, so their behavior is byte-identical. A 5090
   boot that derives a budget above the floor (a smaller model, a shorter context or lower caps) takes the partial
   pass, and no 5090 cell measured that.
2. **The pass.** Free the arrival's shortfall, the entry the arrival will itself publish, and what the sessions
   still priming will allocate, oldest entry first, and keep the rest. The shortfall is read unbooked (the booked
   reading floors at 0 and would understate it). The arrival's entry is bounded by a whole-prompt entry
   (`model_prefix_entry_bytes` at its prompt length, 0 under the seed floor): admission's charge does not hold it,
   and after a partial pass only the transient reserve would. The owed bytes are the prime transients
   (`pending_prime_bytes_v2`: one call's slab growth and returned rows, the unstarted walkers' stacks, the
   checkpoints the primes will take) and the prefix entries the priming sessions will publish
   (`pending_seed_bytes`). The seeds are counted uncapped: before the pass a full cache replaces bytes when a seed
   lands, but the pass makes room, and the seeds regrow into it. Eviction is `evict_to_bytes`, the cache's one victim
   order (plain LRU, `PREFIX-CACHE-POLICY.md`); it never takes a leased entry, so the arrival's donor and the
   restorable spare's pins stay. Then the settle step (memra#346) hands the bytes to the driver, and the headroom is
   read again.
3. **The fallback.** If the arrival is still short (allocator rounding, a neighbour's growth between readings, or
   leased bytes the pass could not reach), the rest of the unleased cache goes, as before. The restorable spare's
   climb and the parked-session ladder then run unchanged, on the empty evictable set they assume.
4. **The booking.** The flush hid a gap in the door-off gate: it books nothing for admitted sessions that are still
   priming. A partial pass leaves free only the arrival's need plus what the priming sessions owe, so the next
   arrival in the same burst would pass the gate on the owed bytes. Each arrival's charge includes a full 2.3 GB
   prefill workspace, but the shared prime slab means it allocates about 1 GB at admission, so several pass before
   any defers. Their seeds then land into the room the pass made: in the limit cell's burst, about 5.5 GB of seeds
   for the six sessions priming at the defer and 11 GB for all twelve, against a few GB of free memory. So once a
   pass has freed bytes, admission books what the pass counted as owed: the prime transients, and the pending seeds
   capped at the room the cache has left (`seed_booking_cap`, the door's own seed term; on a full cache a seed
   replaces bytes and books 0). A later arrival of the burst then defers and runs its own pass. The booking stays
   armed through a fallback flush (after a flush its seed term binds only when free memory is short) and ends at an
   arrival that finds nothing owed: no session priming and no seed pending.
5. **One line per pass that freed something.** A deferred arrival comes back through the reclaim every tick, so a
   pass that evicted nothing prints nothing, as the flush did.
6. **No flag.** The rollback is the revert.

The lane's banked code (`bank/prefix-budget-gds-20261107`, 39ec446dd5: `reclaim_prefix_target`,
`reclaim_owed_bytes`) had the pass and the owed term, on an older reclaim site with the off-tick and demote arms. It
had no booking, no arrival entry and applied on every discrete card. This change re-derives the pass against the
current site and adds the floor key, the arrival's entry and the booking. The arrival's entry, the booking's prime
term and its life through a fallback came from the independent review of the first version (24f0f95825), whose
acceptance run was stopped in its first B arm so that the measured binary is the final code.

## Results (one RTX PRO 6000 Max-Q, 2026-11-09): pass

A = #1090's head 6a9db1447d (the flush; main's code since #1090 landed), B = d6c81127e7 (this change). Release builds of
both from one clone and one target dir (sha256 `fdc65efbd4d2c716...` and `d2b96ff49a940906...`, in each `arm.txt`).
Four fresh boots interleaved A B B A on one card, then the mixed shape A B B A A B on the same card, 280 W, no thermal
throttle (max 91 C, fans at most 91 %), every process on 16 of the box's 64 threads. Untimed: the box's other cards
ran another lane's jobs throughout. Tables: `harness/reclaim_table.py` (`pro6000-maxq/limit-table.md`,
`mixed-table.md`); campaign scripts and power logs in `pro6000-maxq/campaign/`.

### The at-the-limit cell (`pro6000-maxq/limit/`, `harness/limit.py` from the #1090 lane)

| arm | budget MB | reclaims (partial, still short) | entries / GB freed (largest) | booking on / off | evictions preflight / LRU | OOM lines | pressure: errors, TTFT p50 / max s | repeats: hits, cached tokens, TTFT p50 s | greedy texts = A |
|---|---|---|---|---|---|---|---|---|---|
| limit-1-A | 54277 | 1 (0, 0) | 57 / 54.2 (57 / 54.2) | 0 / 0 | 7 / 0 | 0 | 4 of 12, 52.5 / 87.4 | 0 of 16, 0, 10.46 | 92/92 |
| limit-2-B | 54277 | 6 (6, 0) | 19 / 18.2 (8 / 7.8) | 1 / 1 | 7 / 0 | 0 | 4 of 12, 52.0 / 88.1 | 16 of 16, 425984, 0.12 | 92/92 |
| limit-3-B | 54277 | 6 (6, 0) | 19 / 18.2 (8 / 7.8) | 1 / 1 | 7 / 0 | 0 | 4 of 12, 52.0 / 88.0 | 16 of 16, 425984, 0.12 | 92/92 |
| limit-4-A | 54277 | 1 (0, 0) | 57 / 54.2 (57 / 54.2) | 0 / 0 | 7 / 0 | 0 | 4 of 12, 52.6 / 87.5 | 0 of 16, 0, 10.47 | 92/92 |

- **The flush is gone.** On B every arrival from the seventh on deferred and ran its own partial pass: 8, 2, 3, 2, 2
  and 2 entries, 18.2 GB in all, each covered on the re-read. No fallback flush ran. The cache kept 38 of its 57
  entries (36.0 GB); the oldest fill prompts went. A flushed 57 entries (54.2 GB) at the seventh arrival on both
  arms, as in #1090's cell. The first pass freed 7,768 MB against a need of 7,553 MB, of which 7,114 MB was owed
  (six priming sessions' seeds and transients, and the arrival's own entry); each later pass freed one arrival's
  share. Every pass is listed in `limit-table.md`. The design note expected 15 to 25 GB over the burst.
- **The repeats hit.** B: 16 of 16, 425,984 of 426,304 prompt tokens cached, TTFT p50 116 ms against 10.46 s on A.
  The repeat phase took 17 s against 183 s.
- **Same texts.** All 92 compared greedy rows match A on both B arms, the 16 repeats B restored from the cache
  included (A primed them cold).
- **No OOM.** No `out of memory` line, no `snapshot failed`, no step-OOM reclaim on any arm. The device free floor
  is 4,968 MiB on every arm (the fill phase). The booking armed at the first pass and ended once the burst had
  landed (one on and one off line per B arm).
- **Pressure TTFT.** By rank, B against A (two arms each, averaged): -0.18, -0.26, -0.33, -0.43, -0.53, -0.63,
  -0.73 and +0.55 s; the two A arms agree within 0.10 s at every rank. p50 52.0 s against 52.5 to 52.6 s, max 88.0
  to 88.1 s against 87.4 to 87.5 s, mean 47.5 s against 47.8 s. The rank-8 request shares its prefill with the four
  that miss the 90 s first-token deadline (HTTP 408 on every arm, as in #1090's cell), and which request ends up there
  differs (pressure-6 on A, pressure-7 on B): B admits arrivals 7 to 12 one pass at a time, A all at once after its
  flush. The fill phase, where neither arm reclaims, also reads B 0.15 to 0.24 s under A at p50, so differences
  under a second here sit near what the two binaries show without the mechanism.

### Regression: the mixed shape (`pro6000-maxq/mixed/`, counts only)

| arm | budget MB | repeats hit | repeat TTFT median ms | batch tok/s | batch TTFT p50 ms | reclaims (partial) | evict lines | OOM lines | greedy texts = A |
|---|---|---|---|---|---|---|---|---|---|
| mixed-1-A | 54277 | 6 of 6 | 443 | 220.78 | 598 | 0 (0) | 0 | 0 | 131/131 |
| mixed-2-B | 54277 | 6 of 6 | 265 | 225.84 | 587 | 0 (0) | 0 | 0 | 127/127 |
| mixed-3-B | 54277 | 6 of 6 | 193 | 224.94 | 588 | 0 (0) | 0 | 0 | 127/127 |
| mixed-4-A | 54277 | 6 of 6 | 447 | 226.58 | 596 | 0 (0) | 0 | 0 | 130/130 |
| mixed-5-A | 54277 | 6 of 6 | 403 | 227.02 | 588 | 0 (0) | 0 | 0 | 130/130 |
| mixed-6-B | 54277 | 6 of 6 | 188 | 225.32 | 595 | 0 (0) | 0 | 0 | 127/127 |

No reclaim ran on any arm: the cache never filled (as in #1090's mixed cells), so the change is not reached. Every
arm hit 6 of 6 repeats with no eviction and no OOM line, and every compared greedy row matches A. The repeat TTFT is
bimodal per request (about 185 ms, or about 700 ms when a batch prefill is in flight), and the spec scheduler's
cohort choices vary per boot on both arms (mixed-4-A demoted a cohort to batched decode, mixed-6-B did not; #1090's
own B arms on one build split the same way), so the medians in that column carry no A/B signal.

### The first version's stopped run (`pro6000-maxq/stopped-24f0f95825/`)

The campaign of the first version (24f0f95825) ran one A arm and was stopped 90 seconds into its first B arm, after
the independent review found the gaps d6c81127e7 closes (the arrival's own entry, the booking's prime term). Its A arm
reads like the other two: one flush of 57 entries, 0 of 16 repeats, pressure 52.0 / 88.0 s. The B arm has only its
boot and the start of its fill, and nothing is concluded from it.

### Scope

One card, one model (Qwen3.8-27B NVFP4 + Q5_K with its MTP head), one served context (32,768), one cap
(`MEMRA_MAX_SESSIONS=12`), two shapes. Unmeasured: an RTX 5090 boot that derives a budget above the floor, and other
burst shapes at the limit. Unchanged by construction: the floor, unified memory (GB10), per-device readings,
operator budgets and the door-armed path.

## Acceptance plan (one RTX PRO 6000 Max-Q, untimed)

- Arms: A = #1090's head (the flush), B = this change. Interleaved A B B A, `harness/limit.py` through the lane's arm
  script, `MEMRA_CTX=32768`, `MEMRA_MAX_SESSIONS=12`.
- Pass: B's repeats hit most of the 16; no OOM line; pressure TTFT not worse than A beyond noise; the flush gone or
  rare (every reclaim line counted with what it freed).
- Regression: the RTX 5090 lane's mixed shape (`harness/mixed.py`, `MEMRA_MAX_SESSIONS=12`) on the same card class,
  A against B, three runs per arm, counts only.
- A negative result keeps the code on a research branch and is written up here.
