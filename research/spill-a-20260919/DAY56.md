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
