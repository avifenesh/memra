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
   pass has freed bytes, admission books what the priming sessions owe (the prime transients and the pending seeds),
   capped at the room the cache has left. The passes freed at least that room, so the booking never keeps more than
   they freed, and as landing seeds refill the cache it falls to 0: steady traffic on a full cache meets the
   unbooked door-off gate again. A later arrival of the burst defers and runs its own pass. The booking stays armed
   through a fallback flush (the room is then the whole budget, so everything owed is booked, up to what the flush
   freed) and ends at an arrival that finds nothing owed: no session priming and no seed pending. The flag is
   worker-wide and counts every model's sessions; like the door's own booking it reduces the primary device's
   reading. While it is armed, the restorable spare's climb and the parked-session ladder read the booked headroom,
   so an arrival the cache cannot cover may evict parked sessions for what other sessions owe, as under the door.
5. **One line per pass that freed something.** A deferred arrival comes back through the reclaim every tick, so a
   pass that evicted nothing prints nothing, as the flush did.
6. **No flag.** The rollback is the revert.

The lane's banked code (`bank/prefix-budget-gds-20261107`, 39ec446dd5: `reclaim_prefix_target`,
`reclaim_owed_bytes`) had the pass and the owed term, on an older reclaim site with the off-tick and demote arms. It
had no booking, no arrival entry and applied on every discrete card. This change re-derives the pass against the
current site and adds the floor key, the arrival's entry and the booking. The arrival's entry, the booking's prime
term and its life through a fallback came from the independent review of the first version (24f0f95825), whose
acceptance run was stopped in its first B arm so that the measured binary is the final code. The cap of the whole
booking at the cache's room came from the second review (of d6c81127e7): without it the booking would hold every
prime transient for as long as some session primes, which on steady traffic is the whole busy period.

## Results (one RTX PRO 6000 Max-Q card class, 2026-11-09): pass

The final code is ab21e254a6; the head after it (52c1389828) changes one log string. A = main d35337f51a (#1090
landed: the flush), B = ab21e254a6. Release builds of both from one fresh clone and one target dir (sha256
`5274863eec2ecd95...` and `a00d86e9b365e46c...`, in each `arm.txt`). Every arm is a fresh boot at
`MEMRA_CTX=32768`, `MEMRA_MAX_SESSIONS=12`, 280 W (no thermal throttle: max 91 C, fans at most 92 %), every process
on 16 of the box's 64 threads. Each A/B pair ran on one card. Untimed: the box's other cards ran another lane's jobs,
and from 10:06 this lane's second card ran beside the first. The waves cell's later pairs ran at 250 W (below). Tables: `harness/reclaim_table.py`; campaign scripts,
power and thermal logs under `pro6000-maxq/*/campaign/` and `pro6000-maxq/campaign/`.

### The at-the-limit cell (`pro6000-maxq/limit/`, GPU 0, A B B A, `harness/limit.py` from the #1090 lane)

| arm | budget MB | reclaims (partial, still short) | entries / GB freed (largest) | booking on / off | evictions preflight / LRU | OOM lines | pressure: errors, TTFT p50 / max s | repeats: hits, cached tokens, TTFT p50 s | greedy texts = A |
|---|---|---|---|---|---|---|---|---|---|
| limit-1-A | 54277 | 1 (0, 0) | 57 / 54.2 (57 / 54.2) | 0 / 0 | 7 / 0 | 0 | 4 of 12, 52.2 / 88.3 | 0 of 16, 0, 10.37 | 92/92 |
| limit-2-B | 54277 | 6 (6, 0) | 19 / 18.2 (8 / 7.8) | 1 / 1 | 7 / 0 | 0 | 4 of 12, 52.5 / 87.7 | 16 of 16, 425984, 0.12 | 92/92 |
| limit-3-B | 54277 | 6 (6, 0) | 19 / 18.2 (8 / 7.8) | 1 / 1 | 7 / 0 | 0 | 4 of 12, 52.8 / 87.9 | 16 of 16, 425984, 0.12 | 92/92 |
| limit-4-A | 54277 | 1 (0, 0) | 57 / 54.2 (57 / 54.2) | 0 / 0 | 7 / 0 | 0 | 4 of 12, 52.4 / 87.3 | 0 of 16, 0, 10.44 | 92/92 |

- **The flush is gone.** On B each arrival from the seventh on deferred and ran its own partial pass: 8, 2, 3, 2, 2
  and 2 entries, 18.2 GB in all, each covered on the re-read, no fallback. The cache kept 38 of its 57 entries
  (36.0 GB); the oldest fill prompts went. A flushed 57 entries (54.2 GB) at the seventh arrival on both arms, as in
  #1090's cell. The first pass freed 7,768 MB against a need of 7,553 MB, of which 7,114 MB was owed (six priming
  sessions' seeds and transients, and the arrival's own entry); each later pass freed one arrival's share
  (`limit-table.md` lists them). The design note expected 15 to 25 GB over the burst.
- **The repeats hit.** B: 16 of 16, 425,984 of 426,304 prompt tokens cached, TTFT p50 116 to 117 ms against 10.37 to
  10.44 s on A. The repeat phase takes 17 s against 183 s.
- **Same texts.** All 92 compared greedy rows match A on both B arms, the 16 repeats B restored from the cache
  included (A primed them cold).
- **No OOM.** No `out of memory` line, no `snapshot failed`, no step-OOM reclaim on any arm. The device free floor
  is 4,936 MiB on B and 4,968 MiB on A, both in the fill phase, where neither arm reclaims. The booking armed at the
  first pass and ended once the burst had landed (one on and one off line per B arm). The room cap never bound here:
  the passes are the same, byte for byte, as in the run before the cap (below).
- **Pressure TTFT is within run-to-run variation.** By rank, B against A (two arms each, averaged): 0.00, +0.10,
  +0.20, +0.30, +0.38, +0.59, +0.65 and +0.02 s; the A arms themselves spread up to 0.32 s at ranks 1 to 7 and
  1.0 s at rank 8. The run before the cap, with the same six passes, read B 0.18 to 0.73 s faster at ranks 1 to 7 and
  0.55 s slower at rank 8. The sign flips between runs whose reclaim work is identical, so these sub-second
  differences are not the mechanism. Four of twelve pressure requests miss the 90 s first-token deadline (HTTP 408)
  on every arm, as in #1090's cell.

### Repeated pressure waves (`harness/limit_waves.py`): one A/B pair per card and power regime

The same fill, then three pressure waves of 12 fresh unique prompts back to back, then the 16 repeats
(`harness/limit_waves.py`; tables `pro6000-maxq/waves-*/*/waves-table.md`). Each A/B pair ran on one card in one power
regime. The first pair ran at 280 W on GPU 1. At 10:58 the box's GPUs 0 and 1 went back to their 250 W cap while
another lane's timed blocks ran: the two arms then running were stopped (kept as `.stopped`, out of the tables), and
one pair per card was run again at 250 W with no power raise. GPU 0's 280 W B arm (`waves-280w/gpu0/`) has no A
partner and is shown alone.

280 W, GPU 1 (`waves-280w/gpu1/`):

| arm | reclaims (partial, still short) | entries / GB freed (largest) | booking on / off | OOM lines | pressure waves: errors, TTFT p50 / max s | repeats: hits, cached tokens, TTFT p50 s | greedy texts = A |
|---|---|---|---|---|---|---|---|
| waves-1-A | 1 (0, 0) | 57 / 54.2 (57 / 54.2) | 0 / 0 | 0 | 5, 46.9 / 88.3; 5, 46.7 / 88.1; 5, 46.8 / 88.1 | 0 of 16, 0, 10.56 | 116/116 |
| waves-2-B | 7 (7, 0) | 36 / 34.3 (15 / 14.2) | 2 / 2 | 0 | 5, 46.8 / 88.2; 5, 46.7 / 88.4; 5, 46.8 / 88.2 | 16 of 16, 425984, 0.12 | 114/116 |

250 W, GPU 1 (`waves-250w/gpu1/`) and GPU 0 (`waves-250w/gpu0/`):

| arm | reclaims (partial, still short) | entries / GB freed (largest) | booking on / off | OOM lines | pressure waves: errors, TTFT p50 / max s | repeats: hits, cached tokens, TTFT p50 s | greedy texts = A |
|---|---|---|---|---|---|---|---|
| waves-1-B | 7 (7, 0) | 36 / 34.3 (15 / 14.2) | 2 / 2 | 0 | 5, 49.8 / 88.6; 5, 49.9 / 88.7; 5, 49.9 / 88.6 | 16 of 16, 425984, 0.12 | 114/116 |
| waves-2-A | 1 (0, 0) | 57 / 54.2 (57 / 54.2) | 0 / 0 | 0 | 5, 49.9 / 88.6; 5, 49.9 / 88.8; 5, 49.8 / 88.6 | 0 of 16, 0, 11.28 | 116/116 |

| arm | reclaims (partial, still short) | entries / GB freed (largest) | booking on / off | OOM lines | pressure waves: errors, TTFT p50 / max s | repeats: hits, cached tokens, TTFT p50 s | greedy texts = A |
|---|---|---|---|---|---|---|---|
| waves-1-A | 1 (0, 0) | 57 / 54.2 (57 / 54.2) | 0 / 0 | 0 | 5, 49.5 / 89.4; 5, 49.4 / 89.4; 5, 49.4 / 89.3 | 0 of 16, 0, 11.09 | 116/116 |
| waves-2-B | 7 (7, 0) | 36 / 34.3 (15 / 14.2) | 2 / 2 | 0 | 5, 49.4 / 89.0; 5, 49.3 / 89.2; 5, 49.6 / 89.5 | 16 of 16, 425984, 0.12 | 114/116 |

280 W, GPU 0, unpaired (`waves-280w/gpu0/`):

| arm | reclaims (partial, still short) | entries / GB freed (largest) | booking on / off | OOM lines | pressure waves: errors, TTFT p50 / max s | repeats: hits, cached tokens, TTFT p50 s | greedy texts = A |
|---|---|---|---|---|---|---|---|
| waves-1-B | 8 (8, 0) | 36 / 34.3 (13 / 12.3) | 2 / 2 | 0 | 5, 46.2 / 88.9; 5, 46.2 / 89.2; 5, 46.6 / 87.7 | 16 of 16, 425984, 0.12 | 0/0 |

- **Across three bursts B never flushed and never ran short.** Every B arm ran 7 or 8 covered passes freeing 36
  entries (34.3 GB) in all. Wave 1 took six passes, as in the single-burst cell. In wave 2 the first defer came with
  11 sessions priming and one pass freed 15 entries (13 and 2 in two passes on GPU 0's unpaired arm); wave 3
  needed none. The booking armed and ended once per
  wave that reclaimed (two on and two off lines). A flushed 57 entries (54.2 GB) once per arm, in wave 1, and had
  room for waves 2 and 3.
- **The repeats still hit.** B: 16 of 16 on every arm (425,984 cached tokens, TTFT p50 0.12 s). A: 0 of 16 (10.6 to
  11.3 s).
- **No OOM** line on any arm, in either regime.
- **Each wave's TTFT matches A.** At 280 W, p50 46.7 to 46.9 s on both arms and max 88.1 to 88.4 s. At 250 W, p50
  49.3 to 49.9 s and max 88.6 to 89.5 s, B within 0.4 s of its own pair's A in every wave. Five of twelve requests
  per wave miss the 90 s first-token deadline on every arm.
- **Texts.** Every row that completed on both arms of a pair matches. The two rows per B arm that differ are one
  request that missed the deadline on one arm and completed on the other; within each pair they swap places.

### Regression: the mixed shape (counts only), A B B A A B on each card

GPU 1 (`pro6000-maxq/gpu1/mixed/`):

| arm | budget MB | repeats hit | repeat TTFT median ms | batch tok/s | batch TTFT p50 ms | reclaims (partial) | evict lines | OOM lines | greedy texts = A |
|---|---|---|---|---|---|---|---|---|---|
| mixed-1-A | 54277 | 6 of 6 | 291 | 235.04 | 604 | 0 (0) | 0 | 0 | 135/135 |
| mixed-2-B | 54277 | 6 of 6 | 287 | 235.38 | 586 | 0 (0) | 0 | 0 | 135/135 |
| mixed-3-B | 54277 | 6 of 6 | 230 | 228.48 | 616 | 0 (0) | 0 | 0 | 134/134 |
| mixed-4-A | 54277 | 6 of 6 | 387 | 229.71 | 616 | 0 (0) | 0 | 0 | 135/135 |
| mixed-5-A | 54277 | 6 of 6 | 336 | 231.56 | 598 | 0 (0) | 0 | 0 | 135/135 |
| mixed-6-B | 54277 | 6 of 6 | 346 | 230.82 | 600 | 0 (0) | 0 | 0 | 135/135 |

GPU 0, after the acceptance cell (`pro6000-maxq/gpu0/mixed/`):

| arm | budget MB | repeats hit | repeat TTFT median ms | batch tok/s | batch TTFT p50 ms | reclaims (partial) | evict lines | OOM lines | greedy texts = A |
|---|---|---|---|---|---|---|---|---|---|
| mixed-1-A | 54277 | 6 of 6 | 364 | 230.79 | 586 | 0 (0) | 0 | 0 | 135/135 |
| mixed-2-B | 54277 | 6 of 6 | 357 | 231.0 | 584 | 0 (0) | 0 | 0 | 135/135 |
| mixed-3-B | 54277 | 6 of 6 | 366 | 230.62 | 586 | 0 (0) | 0 | 0 | 135/135 |
| mixed-4-A | 54277 | 6 of 6 | 349 | 231.1 | 590 | 0 (0) | 0 | 0 | 135/135 |
| mixed-5-A | 54277 | 6 of 6 | 365 | 230.32 | 592 | 0 (0) | 0 | 0 | 135/135 |
| mixed-6-B | 54277 | 6 of 6 | 343 | 230.9 | 589 | 0 (0) | 0 | 0 | 135/135 |

No reclaim ran on any arm: the cache never filled (as in #1090's mixed cells), so the change is not reached. Every arm
hit 6 of 6 repeats with no eviction and no OOM line, and every compared greedy row matches A. The repeat TTFT is
bimodal per request (about 185 ms, or about 700 ms when a batch prefill is in flight) and the spec scheduler's cohort
choices vary per boot on both arms, so that column carries no A/B signal.

### Earlier runs

- **Before the room cap (`pro6000-maxq/run-d6c81127e7/`, d6c81127e7 against 6a9db1447d, same campaign on GPU 0).**
  The same counts: six covered passes per B arm freeing 18.2 GB, 38 entries kept, 16 of 16 repeats at 116 ms against
  10.46 s, no OOM line, 92 of 92 texts, pressure p50 52.0 s against 52.5 to 52.6 s. Its mixed arms reached no reclaim.
  The second independent review then capped the booking at the cache's room (ab21e254a6), and the cells above were
  rerun on that code.
- **The first version's stopped run (`pro6000-maxq/stopped-24f0f95825/`).** The campaign of 24f0f95825 ran one A arm
  and was stopped 90 seconds into its first B arm, after the first independent review found the gaps d6c81127e7
  closes (the arrival's own entry, the booking's prime term). Its A arm reads like the others: one flush of 57
  entries, 0 of 16 repeats, pressure 52.0 / 88.0 s. The B arm has only its boot and the start of its fill, and
  nothing is concluded from it.

### Scope

One card class, one model (Qwen3.8-27B NVFP4 + Q5_K with its MTP head), one served context (32,768), one cap
(`MEMRA_MAX_SESSIONS=12`), three shapes, two power regimes for the waves (280 and 250 W). An RTX 5090 boot above the
floor is measured below. Steady traffic at the limit beyond back-to-back bursts stays owed until its cell lands. Unchanged by construction: the floor, unified memory (GB10),
per-device readings, operator budgets and the door-armed path.

## An RTX 5090 boot above the floor (2026-11-09): `rtx5090-above-floor/`

The scope above listed this as unmeasured: on the 24 GB RTX 5090 Laptop every measured Qwen3.8-27B configuration
derives the two-entry floor, but a smaller model or lower caps derive more, and then the partial pass applies.

**The config.** Qwen3.5-9B NVFP4 with its MTP head (`Qwen3.5-9B-NVFP4-MTP-GGUF.gguf`), `MEMRA_CTX=32768`. A full-context
entry is 539,230,208 B (14,848 B/token plus 52.7 MB of recurrent state), so the floor is 1,078 MB; one served-context
session is 600,047,616 B; boot driver free is 18.31 GB. The derived budget by interactive cap (`probe/`, A's boot
line; the judge and harvest lanes add 12 resident sessions to every row):

| `MEMRA_MAX_SESSIONS` | resident sessions | session peak | derived budget | against the floor |
|---|---|---|---|---|
| 12 | 24 | 14.40 GB | 1,078 MB | the floor |
| 8 | 20 | 12.00 GB | 2,549 MB | 2.4x |
| 4 | 16 | 9.60 GB | 4,949 MB | 4.6x |
| 2 | 14 | 8.40 GB | 6,150 MB | 5.7x |

The cells run at `MEMRA_MAX_SESSIONS=4`: 4.6 times the floor, about nine full-context entries, a real low-concurrency
server and not the far end of the range. Both arms boot the same budget line (#1090 is on main).

**Arms.** A = main fa0ee88acf. B = #1124 (3448243319) merged onto that main in a scratch worktree (tree 113ef7a1e0,
not pushed), so the arms differ only by #1124; #1124's own base predates main's new GDN prime kernels. Release builds
on the rig (`<bins>` sha256 `45896f1d1143c530...` and `a940b9147037c98b...`). One gpulock lease per arm (`harness/pairs.sh`),
A B B A as two pairs; both pairs ran intact (no other lease between their arms). The card was yielded once for another
lane's record, at an arm boundary before this block; the A arm before that break is kept in `limit-prebreak/`. Both arms
boot the same budget line, 4,949 MB.

### The scaled at-the-limit cell (`limit/`, `harness/limit.py` at `--tok 26000 --fill 32 --fill-conc 4 --pressure 4 --repeat 6`)

The #1090 lane's harness at its own defaults for concurrency (fill four at a time, pressure at the interactive cap),
with the prompt size and counts scaled to this budget: 32 unique fill prompts of about 28.6k tokens (one entry
about 0.48 GB each, nine fill the budget), 4 more at once, then the 6 most recent fill prompts again.

| arm | budget MB | reclaims (partial, still short) | entries / GB freed (largest) | booking on / off | evictions preflight / LRU | OOM lines | pressure: errors, TTFT p50 / max s | repeats: hits, cached tokens, TTFT p50 s | greedy texts = A |
|---|---|---|---|---|---|---|---|---|---|
| limit-1-A | 4949 | 6 (0, 0) | 30 / 15.2 (9 / 4.5) | 0 / 0 | 0 / 1 | 0 | 0 of 4, 29.6 / 29.7 | 0 of 6, 0, 7.86 | 42/42 |
| limit-2-B | 4949 | 14 (12, 0) | 28 / 14.4 (4 / 2.1) | 7 / 7 | 0 / 1 | 0 | 0 of 4, 29.8 / 29.9 | 1 of 6, 28960, 7.64 | 42/42 |
| limit-3-B | 4949 | 16 (13, 0) | 33 / 16.9 (1 / 2.1) | 8 / 7 | 0 / 1 | 0 | 0 of 4, 28.4 / 28.5 | 0 of 6, 0, 7.62 | 42/42 |
| limit-4-A | 4949 | 6 (0, 0) | 25 / 12.8 (9 / 4.5) | 0 / 0 | 0 / 1 | 0 | 0 of 4, 28.9 / 28.9 | 1 of 6, 28960, 7.52 | 42/42 |

| arm | inserts | resident MB at insert: median / mean / max | inserts into an empty cache (resident = the entry) |
|---|---|---|---|
| limit-1-A | 42 | 1609 / 1913 / 4997 | 9 |
| limit-2-B | 41 | 2060 / 2297 / 4997 | 5 |
| limit-3-B | 42 | 1958 / 2196 / 4997 | 6 |
| limit-4-A | 41 | 1609 / 1934 / 4997 | 9 |

- **On this card the reclaim runs through the fill, on both arms.** The 16 parked sessions of about 0.54 GB each
  (8.6 GB, outside the session-peak estimate) outgrow the budget's 4.95 GB, and an arrival of 28.6k tokens needs about
  3.0 GB (a 1,384 MB charge plus the 1.5 GiB transient reserve) while free memory sits at 2.0 to 3.0 GB. A flushed 6
  times per arm (25 to 30 entries, 12.8 to 15.2 GB, up to 9 entries at once). B ran 12 or 13 partial passes per arm,
  every one covered, no fallback (14 or 16 reclaim lines with the ones that took only parked sessions), freeing 28 to 33
  entries (14.4 to 16.9 GB, at most 4 at once). The booking armed and ended 7 or 8 times per arm. From the reclaim
  numbers, a window where the fill fits and only the burst defers needs fill prompts within about 0.5 GB of the limit,
  narrower than the estimate's error, so this regime is what the card reaches at this config.
- **Safe and flat.** No OOM line, every greedy row equal on every arm. Repeats: A 0 and 1 of 6, B 1 and 0 of 6. B held
  more of the cache (median resident at insert 2,060 and 1,958 MB against 1,609 MB) and started from an empty cache less
  often (5 and 6 inserts against 9), but the parked pool keeps the cache to a few entries either way, so the repeats
  did not move. Pressure TTFT p50 29.8 and 28.4 s against 29.6 and 28.9 s. The pre-break A arm and the two sizing arms
  (`sizing1/`, the same arguments; `sizing2/`, the fill one at a time) read the same.

### The mixed shape (`mixed/`, counts only, A B B A)

| arm | budget MB | repeats hit | repeat TTFT median ms | batch tok/s | batch TTFT p50 ms | reclaims (partial) | evict lines | OOM lines | greedy texts = A |
|---|---|---|---|---|---|---|---|---|---|
| mixed-1-A | 4949 | 6 of 6 | 1319 | 273.18 | 985 | 0 (0) | 52 | 0 | 150/150 |
| mixed-2-B | 4949 | 6 of 6 | 1334 | 272.08 | 988 | 0 (0) | 52 | 0 | 149/149 |
| mixed-3-B | 4949 | 6 of 6 | 1336 | 271.99 | 991 | 0 (0) | 52 | 0 | 149/149 |
| mixed-4-A | 4949 | 6 of 6 | 1334 | 271.94 | 990 | 0 (0) | 52 | 0 | 148/148 |

No reclaim on any arm; the cache sits at its budget and plain LRU evicts the same 52 entries on every arm. Counts
are identical.

**Verdict.** On an RTX 5090 boot above the floor the partial reclaim engages, stays safe and is flat: no OOM, every
pass covered, the same repeat hits and TTFT, slightly more of the cache kept. Its gain needs a budget large beside what
one arrival and the parked pool take, as on the RTX PRO 6000 at 54 GB; nothing here argues for keying it off on the
5090.

**Validity after rebase.** These cells ran A = fa0ee88acf against B = #1124 on it. Main then gained lever 7 (#1126),
which changes the F8F4 v3 GEMM's partial last wave (a launch-shape change its commits state is bit-identical, aimed at
GB10's 640-tile grids) and nothing in admission or the prefix cache, so the counts stand as behavior evidence for the
rebased head.

## Acceptance plan (one RTX PRO 6000 Max-Q, untimed)

- Arms: A = #1090's head (the flush), B = this change. Interleaved A B B A, `harness/limit.py` through the lane's arm
  script, `MEMRA_CTX=32768`, `MEMRA_MAX_SESSIONS=12`.
- Pass: B's repeats hit most of the 16; no OOM line; pressure TTFT not worse than A beyond noise; the flush gone or
  rare (every reclaim line counted with what it freed).
- Regression: the RTX 5090 lane's mixed shape (`harness/mixed.py`, `MEMRA_MAX_SESSIONS=12`) on the same card class,
  A against B, three runs per arm, counts only.
- A negative result keeps the code on a research branch and is written up here.
