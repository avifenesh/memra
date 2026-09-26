# WP-B day 44: O11 revised, an exact and fast resume (one prime call from a grid checkpoint captured inside the call, plus an off-path settle)

OWED.md O11. The owner, 2026-09-26, verbatim: "resume vs rewind - i think its not or or question, but more of we didnt
make it right yet". Keep (the parked decoded rows) is fast and flips against cold (24 of 60 on both routes, DAY41 2.2,
DAY43 2.1); the grid rewind is exact and costs TTFT x2 to x15 (DAY41 2.2). Both are workarounds for one defect: a
resumed session's reply rows were written by the decode (or spec verify) program, and a cold prime of the next prompt
computes those same tokens with the prime program. This day registers the design that removes the defect and gates it
on both costs. `MEMRA_RESUME_GRID_REWIND` stays a measurement arm (decide-by 2026-10-09); its fate follows this arm's
reading.

## 1. Pre-registration

Committed and pushed before any day-44 code and before any day-44 cell. Nothing in section 1 changes after a number is
seen; a failed clause is recorded as it reads and a revision is a new, dated addendum pushed before its code.

### 1.1 What the receipts already place

- **Re-priming rows is cheap; a second prime call is not.** At the engine level one call priming 128 rows costs what one
  call priming 65 rows costs (`primepath` at 6,144: 653.1 against 665.7 ms with 48 greedy steps; at 30,720 and K=256,
  352 rows against 65 cost +32 ms; DAY41 2.1). In serving, the rewind arm's resume at 6,144 and G=32 re-primes 64 rows
  and doubles TTFT (69 to 138 ms): its checkpoint arming stops the prime at the boundary, snapshots, and primes the
  rest as a second call. At 30,720 the rewind costs about +80 to +90 ms fixed plus about 0.2 ms per re-primed row
  (DAY41 2.2: +94 ms at 64 rows, +138 ms at 288).
- **The grid law.** A prime split on the GDN chunk grid (32 rows, `gdn_chunk_size`) is bit-identical to the monolithic
  prime; a state at a grid point produced by a prime from an earlier grid point equals the cold prime's state there
  (`grid_align_boundary`; the probe's `rewind: EXACT`). Every prime segment must be at least `PRIME_MIN_T` (16) rows,
  or it is primed tokenwise, a second program.
- **The chunked scan's state pass is sequential over chunks.** In `gdn_scan_chunked`, K1 to K3 compute each chunk's
  factors independently, and the state pass (K4, `gdn_chunk_state_mma` / `gdn_chunk_state_f32`) carries the f32 state
  chunk by chunk. Its result after the first k chunks does not depend on later chunks.

### 1.2 The design (`MEMRA_RESUME_EXACT`, default unset; decide-by 2026-10-10)

Unset: today's program, byte for byte. `1`:

- **(a) Capture inside the call, no split.** Every prime call of a resumable session (plain and Qwen MTP spec, the
  sessions `MEMRA_RESUME_GRID_REWIND` arms today) carries a capture point `g`: the largest grid point with at least
  `PRIME_MIN_T` rows after it in the call and at least `PRIME_MIN_T` rows primed before it in the call
  (`grid_align_boundary_within`). No stop is inserted. Per layer:
  - GDN layers: after the layer's chunked scan, one extra state-pass launch over the call's first `(g - start) / 32`
    chunks writes the f32 state at `g` into the capture (the same kernel on a prefix of the same K1 to K3 buffers; no
    kernel source changes); the conv ring at `g` is a copy of the call's input rows `g - 3 .. g` in the ring's layout.
  - Full-attention layers: nothing (the KV rows below `g` are the call's rows; a restore truncates to `g`).
  - Spec sessions also capture the trunk hidden of row `g - 1` (the MTP seed) and the draft KV length at `g`.
  A call that carries a capture runs the eager per-sequence kernels: the batched-prime and fanout paths are refused
  for it, as an armed checkpoint refuses them today, and the carried-prime graph and the Hopper wgmma state kernel are
  not used for that call (both are byte-identical twins of the eager path by their own contracts). The capture is a
  `CacheSnapshot` at `g` (plus the spec fields), stored as the session's checkpoint, replacing the previous one.
- **(b) Exact resume, one call.** An exact-extension hit (plain `continuation_reuse_index`; spec exact probe, with
  `MEMRA_SPEC_BUDGET_CLAMP` for the public-stream key on the spec route) on an entry whose checkpoint is at `g`
  restores the checkpoint (`restore_cache_checkpoint` / `spec_rewind_to_checkpoint`, the existing restores) and primes
  `prompt[g..]` in one call, which captures the next checkpoint. No stop, no second call.
- **(c) Settle, off the critical path.** After a turn parks, when the worker has been idle for 100 ms with no request
  queued, a settle job restores the entry's checkpoint at `g` and primes `committed[g..S]` in one call, `S` the largest
  grid point with at least `PRIME_MIN_T` committed rows after it. The entry's cache is then at `S` and prime-produced
  (the state is a function of `committed[..S]` under the prime program), its `fed` is `committed[..S]`, and its
  checkpoint is the cache itself. A later resume primes `prompt[S..]` in one call: `committed[S..]` (fewer than 48
  rows) plus the new tokens, the same work as keep plus that tail.
- **Receipts:** `[kv-reuse] exact: <plain|spec> resume from <g|S> of <F> committed rows (priming <R> rows, <settled|
  checkpoint>; model <m>)`; `[kv-reuse] exact: settle <id> <g> -> <S> (<R> rows, <ms> ms)`; `[kv-reuse] exact:
  declined (<why>); cold`.

### 1.3 Race cases and failure, each with its registered outcome

1. **The next turn arrives before the settle starts** (within the grace, or the worker was busy): (b) from `g`, one call,
   exact. The settle for that entry is dropped (the entry is moved out of the pool by the resume).
2. **The next turn arrives while a settle call runs:** the worker is single-threaded; the arrival is admitted at the next
   tick, after the settle lands, and resumes from `S`. Its wait is bounded by one settle call. The entry is read by the
   probe only between ticks, so it is either at `g` or at `S`, never between.
3. **Cancel or client disconnect:** the settle acts on parked entries, never on a request; a disconnect before park
   parks nothing (today's rule) and queues no settle.
4. **Eviction** (admission reclaim, LRU, the pool caps, a memory park): the settle queue holds entry ids, never
   references; an evicted id is skipped when its turn comes (the reclaim queue's discipline, DAY42 addendum E).
5. **A tenant purge:** as eviction; the purge drops the ids of its namespace from the settle queue.
6. **Memory:** a settle call books its prime workspace like an admitted prime (the admission door's pending term when
   armed); it runs only when the reading covers it, else it waits for the next idle tick. An entry that never settles
   stays exact at `g` (case 1's cost).
7. **A settle that fails** (an OOM or engine error): the entry is dropped (its cache is partly rewritten), the failure is
   quoted in a `[kv-reuse] exact: settle failed (<err>); entry dropped` line, and the next turn primes cold (exact).
8. **A prompt shorter than the resume point plus `PRIME_MIN_T`**, or diverging before it: declined, cold.
9. **A capture that cannot be taken** (a call shorter than 32 plus two floors, a lapped SWA ring, a latent cache): the
   previous checkpoint stays; the next resume re-primes from it (exact, bounded by the rows since).

### 1.4 Alternatives weighed (the lead's list), and why this one

- **Decode writing prime-identical rows:** the decode's recurrent update and its M=1 kernels differ from the chunked WY
  scan and the prime's GEMMs by construction; matching them row by row means recomputing each chunk at every step.
  Not chosen.
- **A bounded rewind with a finer grid:** the grid is the cold prime's (32 rows from 0); a finer grid does not match
  cold. Rewinding to the prompt-end grid point is (b) without (a): it pays today's split. Not chosen alone; (a) is what
  removes the split.
- **Settle only (the lead's candidate) without (a):** the settle's own calls end on the grid, so they need no capture,
  but a zero-gap next turn (an agent loop, the RX client) arrives before any settle, and without (a) its checkpoint is
  the previous turn's resume start: every zero-gap turn re-primes all turns since, unbounded (2.1's R2 defect in
  another form). (a) bounds the zero-gap cost to one call of `G + tail + new` rows; (c) removes the `G` when the next
  turn gives the settle time.
- **Stated residual:** under zero gap the `G` reply rows are re-primed on the critical path, one call. At G=256 that is
  about 50 ms of prime compute on the 27B at 30,720 (1.1's slope), which E2 below may read as a regression. If it does,
  the next revision is the overlap arm (the settle running beside the decode on a second stream into a shadow state),
  pre-registered then from these receipts; it is not built in this day.

### 1.5 CPU before the cards

- Unit tests: the capture point rule (floors, a short call, a call starting on the grid); the settle point rule; the
  settle queue (skip evicted and purged ids, one job per entry, a resume drops its entry's job).
- Census tests: the door is read at the two arming sites, the two exact hits, the settle queue's drain and the idle
  wait, and nowhere else; unset, nothing changes.
- GPU tests (the 5090, `#[ignore]`, under the lock): a capture at `g` equals a split prime's state at `g` bitwise (every
  GDN layer's f32 state and conv ring, every full-attention KV row below `g`), on the 9B and the plain and spec prime
  paths; a resume from the capture equals a cold prime of the same prompt (logits bitwise).

### 1.6 The cells (`day44-client.py` = `day41-client.py` plus a `--turn-gap-ms` argument, one boot = one arm)

- **Shapes:** RX (zero gap, DAY41 1.3 unchanged) and RXg (RX with a 1,000 ms sleep before turns 2 and 3, so a settle can
  land). Lengths 6,144 and 30,720 on both cards plus 122,880 on the target card; G in {32, 256}; N = 5 per (L, G).
- **Arms:** `keep` (both doors unset) and `exact` (`MEMRA_RESUME_EXACT=1`), one binary, both orders, per route and shape;
  the spec route runs both arms with `MEMRA_SPEC_BUDGET_CLAMP=1` so both resume every turn (DAY43 2.1) and the TTFT
  comparison is resume against resume. `MEMRA_PREFIX_CACHE_MB=0` (DAY41 addendum A). Plus `offprev` (the tip with the
  door's commits reverted, plain route, RX at 6,144). Plus the fault boot `exact-settle-fail`
  (`MEMRA_RESUME_EXACT_FAULT=settle-fail`, a forced settle error, plain RXg at 6,144).
- **Stage-0 reading in the same boots:** `MEMRA_TTFT_TRACE=1` on every boot, so each resumed turn's prime and non-prime
  phases are placed.
- Boots per card: 2 routes x 2 shapes x 2 arms x 2 orders = 16, plus `offprev` and the fault boot.

### 1.7 Clauses (the owner's gate is E1 and E2)

- **E1 exactness.** On `exact`, every resumed turn equals its cold twin (0 flips), both routes, both shapes, every L and
  G, both orders, both cards; every resume prints its `exact:` line.
- **E2 speed.** Per route, shape, L and G, same window, both orders: resumed-turn TTFT p50 on `exact` at most 1.05 x
  `keep`'s, and resumed-turn E2E p50 at most 1.05 x `keep`'s. (1.05 is the lane's serving tolerance, as DAY37's A3;
  the owner's words are "no TTFT regression".)
- **E3 resumes.** On `exact`, at least 95% of turns 2 and 3 resume, both routes, both shapes.
- **E4 settle.** On RXg `exact`, at least 80% of resumed turns resume from a settled point (`settled` in the line); on
  the fault boot every forced failure prints its line, the entry is dropped, the next turn is cold and equals its twin.
- **E5 health.** No `CUDA_ERROR_OUT_OF_MEMORY` line, no 503, no crash line on any boot.
- **E6 door OFF.** Every `keep` row at 6,144 on the plain route RX equals `offprev`'s.

Readings, no bound: resumed-turn TTFT p50 and p95 per arm with the trace's prime phase; re-primed rows per resume;
generated tokens over boot wall; idle driver free and pool-cached bytes (the checkpoint bytes per parked session); the
settle count and its wall. Every median states N and the 250 ms regime.

### 1.8 What the reading decides

Nothing moves a default in this day. If E1 to E6 pass on a card class, the arm is PROMOTE-ELIGIBLE there and the grid
rewind door has no remaining use (its deletion is then owed under door hygiene). If E1 passes and E2 fails at some
(route, shape, L, G), the arm is exact but slower there: the overlap revision of 1.4 is owed, from those receipts. If E1
fails, the cause is quoted and the arm is revised under a new addendum.

### 1.9 Price

Code: about 3 agent-days (the in-call capture on the plain and spec prime paths with its GPU tests, about 1.5; the door,
the exact hits, the settle queue and its race handling, about 1; the client, runner and reader, about 0.5). Cells: the
5090 about 5 h, the target card about 8 h.

### 1.10 Addendum A (2026-09-26, the implementation as built, before any cell)

No clause, bound or reading changes. What the code settled that section 1 left open:

- **The capture:** `memra_engine::grid_capture` (armed per call by a drop guard, collected after it), the extra state
  pass in `gdn_scan_chunked_capture` (mma and f32 pairs; the Hopper fused arm takes none), the ring in
  `ssm_conv_ring_capture`, the call-boundary case in `prime_chunk`. The spec capture runs on the cooperative MTP walker
  only (the served default); without the walker a spec session takes no in-call capture and its next resume declines
  cold.
- **GPU checks already run on the 5090 (the 9B, `rtx5090-day44/gpu-tests/`):** the scan capture equals the prefix
  scan's state bitwise on both pairs; a capture inside one call equals the split prime's state at the point; a resume
  from it, and a settle (its own call, with the request end unknown) then a resume, give the cold prime's logits
  bitwise.
- **The settle queue** runs newest parked entry first and keeps at most 64 jobs; a settle's zero-decode spec burst
  uses K = `MEMRA_SPEC_K` or 3.
- **`offprev`** is the tip with every DAY44 code commit reverted (`day44-nodoor.patch`, crates only).
- **Price, corrected:** 16 serving boots plus two on the target card take about 12 h (the seventh sitting's 9 boots took
  6.4 h), not the 8 h of 1.9; the 5090 about 5 h.

### 1.11 Addendum B (2026-09-26, after a local smoke, before any registered cell)

A smoke on the 5090 (`rtx5090-day44-smoke/`, the 9B at 6,144 tokens, N=5, one order; not a registered cell) placed one
defect: every spec settle failed its row check (`settle failed (settle left 6145 committed rows, expected 6144)`),
because a zero-round spec burst still feeds the boundary token (the init feed, a T=1 decode row) and commits it. The
spec settle is now `spec_prime_settle`: the MTP walker's trunk and draft fill only, with no boundary token, no init
feed and no draft preparation, so every committed row is a prime-program row (`7a4abb4c9`). Addendum A's "zero-decode
spec burst" line is replaced by this. The same smoke read, before the fix: plain RX exact resumed 20 of 20 with 0 flips
against cold (keep: 20 of 20, 12 flips), TTFT p50 46.3 against 45.0 ms at G=32 and 104.2 against 45.2 ms at G=256; plain
RXg exact resumed 20 of 20 from settled points with 0 flips, TTFT p50 42.4 and 43.0 ms (keep RX 45.0 and 45.2 ms); 56
settles of 32 rows at about 38 ms each. No clause, bound or reading changes; the cells run on the fixed binary.

On the fixed binary (`rtx5090-day44-smoke/r2/`, spec route under the clamp, same smoke scale): RX exact resumed 20 of
20 with 0 flips against cold (keep, the first smoke: 20 of 20, 12 flips), every resume from a settled point, TTFT p50
102.9 against keep's 99.2 ms at G=32 and 138.1 against 107.2 ms at G=256 (the client's own gap let the settles start,
and a G=256 settle was still running when the next turn arrived); RXg exact resumed 20 of 20 from settled points with
0 flips, TTFT p50 52.2 and 53.7 ms.

## 2. Results

Written after the runs. Section 1 is unchanged.
