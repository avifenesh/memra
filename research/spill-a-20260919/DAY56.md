# WP-A day 56: OWED item 23, F1's suite cost (the admission-counter tests isolated without serializing them)

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`, on `621502678` plus one attribute commit (origin/main `968c0fa68`
merged: integ63's review fix to `PendingContractDemote`'s doc comment, and the `*.patch -whitespace` marker for
`pro-single-p2/`). CPU only, this rig, every command under `systemd-run --user --scope`.

## 1. Pre-registration (committed before any code)

**The cost** (DAY53 section 7): F1 made `admission_counters_guard()` take `drain_lock()` first, so the seven
queue-bound and wait-ceiling tests and the ten route tests that hold it now wait for every `drain_lock()` holder (26
handler tests, some of them hundreds of milliseconds long). The suite's median `finished in` went from 6.47 s (A',
N=200) to 7.94 s (F1, N=400).

**What each holder needs, read from the code.**

- The seven shed and ceiling tests (`the_queue_bound_sheds_..`, `admission_sheds_only_when_..`,
  `deadline_shed_is_interactive_only_..`, `a_saturated_queue_with_free_http_slots_..`, and the three
  `the_queue_wait_ceiling_..`) exercise the bound's arithmetic on a backlog they set themselves. They need their own
  counters, not an order against anyone.
- The ten route tests register routes under their own model names (`t501-..`, `t503-..`) that no handler test serves,
  and they reach the hybrid lane only through `reserve_interactive_through_contention`, which retries through the
  transient contention a handler request makes. They need order only against the tests that set the global backlog.
- `pending_admission_reservation_is_atomic_and_rolls_back_on_drop` asserts the process-global `PENDING_ADMITS` gauge
  that every handler request moves, `admission_reservations_are_lane_scoped` stores the global lane counters, and
  DAY53's cell D swaps the global interactive backlog to the bound: these three need F1's order against the handler
  tests.

**Design F2.**

1. `reserve_pending_admit_on(st, lane, rl, deadline, ceiling_s, lanes)`: the reservation path with the lane counters
   it reads and takes as a parameter (`&'static [AtomicUsize; 3]`); `PendingAdmissionGuard` keeps the reference and
   releases its lane slot to those counters. `reserve_pending_admit` and `reserve_pending_admit_with_ceiling` pass
   `&worker::ADMISSION_RESERVATIONS`, so production reads, takes and releases exactly the counters it did. The worker's
   own releases (after `commit`) are unchanged.
2. The seven shed and ceiling tests run on their own counters (a leaked fresh `[AtomicUsize; 3]` each) through
   `reserve_pending_admit_on`, and take no guard.
3. `admission_counters_guard()` goes back to its own lock alone (before F1), held by the route tests; a new
   `global_counter_writer_guard()` takes `drain_lock()` then that same lock, held by the three global writers above.
   So the global writers are ordered against both the handler tests and the route tests; the route tests against the
   global writers; the seven isolated tests against nobody.

**Acceptance, stated before any code.**

- (a) CPU cells green; the census, replacing DAY53's: `global_counter_writer_guard()` takes `drain_lock()` before the
  counters' lock, `admission_counters_guard()` takes only the counters' lock; every test that swaps, stores, or
  fetch-adds `worker::ADMISSION_RESERVATIONS` or `worker::PENDING_ADMITS` takes `global_counter_writer_guard()`; no test
  takes `drain_lock()` together with either guard; the production entries pass the global counters; the guard releases
  to the counters it reserved on. Its teeth: one global writer given `admission_counters_guard()` instead fails it.
- (b) 400 full suites in arm A's shape (default threads, `CPUQuota=1200%`): the target of item 21 and its three
  siblings 0 failures, no handler test answering 429 (F1's clause, unchanged), and the suite's median `finished in` at
  most 6.70 s.
- (c) 100 full suites in arm B's shape (`--test-threads 48`, `CPUQuota=400%`): none of item 21's or item 22's tests
  fail; every other red recorded by name.

**The rule.** F2 lands if (a) to (c) hold; otherwise it is reverted in one commit with its runs banked and F1 stays.
No bound moves.

**Budget.** 0.3 agent-day: the code and census 0.1, the 500 suites 0.2.

## 2. F2 as built (`10b9329cc`), and the suites (`day56/suite-{a,b}/`)

- Built as section 1 states. `reserve_pending_admit_on` carries the lane counters (`lanes_counters`), the
  `PendingAdmissionGuard` keeps them (`lanes`) and releases through `worker::release_admission_reservation_on`; the
  route-bound guard names the global counters and never releases a lane slot. The seven shed and ceiling tests call
  `reserve_own`, `reserve_own_with_ceiling` and `reserve_own_interactive` over `own_lane_counters()` and take no guard.
  `admission_counters_guard()` is the counters' lock alone again; `global_counter_writer_guard()` is F1's pair, held by
  `pending_admission_reservation_is_atomic_and_rolls_back_on_drop`, `admission_reservations_are_lane_scoped` and cell D.
- The census `day56_the_admission_writers_are_ordered_against_the_handler_readers` replaces day 53's (its body lookup
  bounded at each fn's closing brace, after a first draft read the next item's doc comment as test code and flagged
  `a_stream_is_immune_to_the_deadline_after_its_first_token`); teeth checked (the lane-scoped test given the plain
  guard: `.. writes a process-global admission counter without the writer guard`).
- CPU cells: server lib `930 passed; 0 failed; 25 ignored`; clippy `-D warnings`; fmt; `git diff --check`.
- **(b), 400 full suites in arm A's shape** (the test binary `4ac9aaa939f68409`, run from `crates/memra-server`):
  `398 rc=0`, `2 rc=101`. `responses_carry_rate_limit_headers_and_slot_frees`, `same_effort_value_..` and
  `same_omitted_request_..` green in 400 of 400; no handler test answered 429 (no `left: 429` in any run). The suite's
  median `finished in` **6.62 s** (N=400, min 6.48, max 9.25) against the bound 6.70 s. The two reds are item 24's
  (`timed out (3000ms) waiting for: yield to T`, run 179) and item 25's (`(0, 1, 0)` against `(0, 0, 0)`, run 156).
- **(c), 100 full suites in arm B's shape:** `100 rc=0`; none of item 21's or item 22's tests red; median 5.56 s.
- A reading on the comparison: the suite grew by nine tests between A' (921) and F2 (930, main's integ63 included), so
  6.62 s is an upper side of the like-for-like cost; F1's 7.94 s and A''s 6.47 s were each one suite's shape.

**Verdict, as registered: F2 passes (a) to (c).** Item 23 closes. Items 24 and 25 now read one red each in arm A's
shape as well (they were arm B's only), which their own items take.

## 3. Addendum, pre-registered (the integ65 review's finding, before any code)

**The hole** (revuto on integ65, #731): F2 isolated the shed tests from `ADMISSION_RESERVATIONS`, but the reservation
path still adds to the process-global `worker::PENDING_ADMITS` on every successful reservation (`lib.rs` around 2918,
and the route path around 3017), and `PendingAdmissionGuard`'s drop releases the global through
`worker::release_pending_admit()`. Four shed tests reserve successfully and now hold no lock
(`admission_sheds_only_when_..`, `deadline_shed_is_interactive_only_..`, `a_saturated_queue_with_free_http_slots_..`,
`the_queue_wait_ceiling_admits_under_it_..`), so their reserve or drop can land inside
`pending_admission_reservation_is_atomic_and_rolls_back_on_drop` (which resets the gauge to 0 and asserts 1, then 0),
and the darklane and admission-yield readers that expect the gauge at 0 can see a stray increment. The day-56 census
missed it: it matched the literal `worker::PENDING_ADMITS` in a test body, and `reserve_own` writes it through the path.

**The fix (F2b), the lead's shape.** One `AdmitCounters` pair (the lane counters and the pending-admits gauge) passed
through `reserve_pending_admit_on` and the route path and kept by the guard, which releases both to the pair it
reserved on; every production entry passes the globals (`AdmitCounters::GLOBAL`), so production reads and writes
exactly the counters it did; the committed path is unchanged (the worker releases the global gauge at its pop). The
shed tests take their own pair (`own_admit_counters()`).

**The census, extended to indirect writers:** every test fn that reserves through a global entry
(`reserve_pending_admit(`, `reserve_pending_admit_with_ceiling(`, `reserve_interactive_through_contention(`) or through
a handler (`chat_completions(`, `completions(`, `messages(`, `responses(`, `embeddings_admitted(`, `rerank_admitted(`,
`chat_completion_admitted(`) holds `drain_lock()`, `admission_counters_guard()` or `global_counter_writer_guard()`;
every test fn that calls `reserve_own` or `reserve_pending_admit_on` passes counters from `own_admit_counters()`; and
the production path names `PENDING_ADMITS` only through `AdmitCounters::GLOBAL`. Tests the extended rule finds calling
a handler without a lock get `drain_lock()` (their handlers may reach a reservation).

**The reproduction and its red arm** (the harness: R3's one-CPU scope with sixteen burners, the group run in one
process with default threads: the pending test beside the four shed tests, `--exact` each, 200 runs). The red arm is the
tree before the fix (the shed tests on their own lanes but the global gauge, `aa325f056`'s code): it must fail the
pending test at least once in 200. The fix: 0 of 200 in the same shape.

**Acceptance.** The red arm reads the race (at least 1 of 200); the fix reads 0 of 200; the census green and its teeth
(one shed test given the global pair fails it; one reserving test without a lock fails it); server lib, clippy and fmt
green. On a miss, F2b is reverted in one commit and the shed tests go back under a lock (the order F1 gave them).

### 3a. The red arm as run, and a deterministic cell added (before any F2b code)

- The group in R3's shape on the tree before F2b (`70be30d39`; the pending test beside the four shed tests, one
  process, 200 runs): **200 of 200 green** (`day56/addendum/red/`). The clause "the red arm reads the race" is **not
  met** by this harness: the tests reserve and drop inside microseconds, and a scheduler-driven overlap did not land in
  200 runs. Recorded as read; the hole is in the code whatever the harness shows (the path writes the global gauge).
- Added, before the fix: `day56_an_isolated_reservation_never_moves_the_global_gauges`, holding
  `global_counter_writer_guard()`: it records the global lane counters and `PENDING_ADMITS`, reserves through
  `reserve_own` on its own pair, and asserts that while the guard lives and after it drops the globals read exactly what
  they did. On the tree before F2b this cell fails at the first read (the gauge one higher); F2b must make it pass. Its
  run on the old tree is the red arm the rule asks for; the 200-run group repeats on the fix as registered.
