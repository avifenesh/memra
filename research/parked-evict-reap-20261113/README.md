# Every eviction a reading or a retry counts on reaps the graveyard first (memra#1278)

Verdict: fixed (memra#1278), on CPU evidence only; no card the push evidence runs on reaches the
code it changes. Under
on-demand KV planes, a dropped parked session's planes wait in the graveyard until a reap, and the
free-memory reading admission and its reclaims take (driver free plus the pool's cached blocks)
leaves the graveyard out. The reclaim-on-defer's parked ladder re-read that reading after each drop
without a reap: the evicted bytes never showed, and one arrival could drop every parked session.
memra#1268 fixed the pre-cohort reclaim's copy of the pattern. Main `df1d23a764` had nine more
eviction sites whose bytes a reading or an allocation retry then counted on, and two re-reads
after drops (the admission drain and the prefix reclaim's settle); each now reaps (`vmm_reap_for`)
before the reading or retry. On a fake device whose freed bytes appear only after a reap, the
ladder, the restorable spare's climb and the one reclaim retry stop after the one eviction that
funds the arrival. With main's program (the reap removed) the ladder spent all three parked
sessions (`left: 3, right: 1`), the climb ran out of parked sessions and dropped the restorable
entry, and the retry failed like the first allocation. Every red arm fails its named test; the
memra-server suite, clippy and fmt pass on the head.

## The instances on main `df1d23a764`

Each row evicts, then reads free memory or retries an allocation that needs the evicted bytes.

| site | what it evicts | what counts on the bytes | the reap now |
|---|---|---|---|
| reclaim-on-defer parked ladder (admission) | globally oldest parked session that frees its planes, one at a time | the headroom re-read after each drop (the issue's line) | `reclaim_parked_ladder` reaps between the eviction and the read (`"reclaim-on-defer"`) |
| restorable spare's climb (admission) | the same, one at a time | the unbooked shortfall re-read after each drop | `reclaim_climb` reaps between them (`"reclaim-on-defer climb"`) |
| park-compact grow (`admit`) | all unleased prefix entries and the oldest parked session | the one cache-allocation retry | `alloc_with_single_reclaim_retry` reaps before its retry (`"park-compact grow"`) |
| plain-affinity grow (`admit`) | the same | the one retry | the same (`"plain-affinity grow"`) |
| cold cache allocation (`admit`) | the same | the one retry | the same (`"reclaim-on-alloc-oom"`) |
| speculative grid rewind and affinity grow (`spec_rewind_oom_reclaim` through `spec_lend_rewind`) | the same | the one retry | the same (`"spec grid rewind"`, `"spec-affinity grow"`) |
| spec pool, learned evict-first (`admit`) | the request's whole spec pool | the session allocation right after | `"spec pool evict-first"` |
| spec pool miss (`admit`) | the request's whole spec pool | the one retry | `"spec pool miss"` |
| admission drain (idle reject) | nothing itself: the graves of the last tick's retires and evictions, which the tick-top reap releases only after admission | the drained re-read after the fence and trim | between the fence and the trim (`"admission-drain"`), as on every reclaim path |
| prefix reclaim settle (`settle_reclaimed_prefix_bytes`) | prefix entries; one that held the last lend of an adopted plane (memra#1221 part 2) sends that plane to the graveyard | the re-read after the settle | after the settle's fence |

Already reaping before this change: the oom-teardown, admin-trim and batch-oom teardowns (DAY37
addendum D, day 49 addendum C), the on-demand grow's reclaim retry, the admission ensure retry and
the pre-cohort reclaim (memra#1266 review 2). The park cap (`prepare_park`) evicts by count and
reads nothing. Unarmed, which is every card but the full-power RTX PRO 6000 Blackwell
Workstation Edition, `vmm_reap_for` returns at once, so the program on every other card is
main's.

## Tests (`crates/memra-server/src/worker.rs`, `parked_evict_reap_1278_tests`)

The fake device: dropping a parked session sends its bytes to the graveyard under on-demand
planes and to the pool's cached blocks under pooled planes; the reading is driver free plus pool
cached; a reap hands the graveyard back to the driver.

- `the_parked_ladder_stops_after_the_one_eviction_its_reap_funds`: 10 GiB free, three parked
  sessions of 4 GiB, an arrival that needs 12 GiB. One eviction, two sessions stay, the order is
  read, evict, reap, read; the same under pooled planes (the reap does nothing there). An arrival
  nothing covers still takes all three; a lost reading stops the ladder on its last reading.
- `the_restorable_climb_stops_after_the_one_eviction_its_reap_funds`: short by 259 MB with a
  249 MB longer entry and 20 MB parked sessions; one eviction leaves 239 MB, and the longer entry
  goes.
- `a_reclaim_retry_gets_the_evicted_bytes_after_its_reap`: 2 GiB free, one 4 GiB parked session,
  a 5 GiB allocation: the retry succeeds.
- `every_eviction_a_reading_or_a_retry_counts_on_reaps_first` (the census): the live code has
  exactly 7 `evict_oldest_parked(` and 2 `evict_oldest_parked_freeing(` calls, each named with
  what follows it, the spec pool's 2 clears and the 3 whole-pool teardowns; every site above
  passes or calls its reap. A new eviction site fails the counts. The memra#1266 census counts 14
  `vmm_reap_for("` sites (it counted 4).

## CPU evidence (`cpu/`)

On the CPU build host (`harness/dd_check.sh`, cores 14 to 31), release profile,
`MEMRA_CUDA_ARCH=120a`, on main `df1d23a764` with the head's `worker.rs` (sha256
`587120d661c80bc4...`, `cpu/summary.txt`):

- the four new tests: 4 passed; the touched tests (the memra#1266 census, the one-retry bound, the
  lend-refused rewind, the lender-skipping ladder, the pre-cohort reclaim's 7, the restorable
  spare's 4): all passed;
- memra-server library suite: 1,266 passed, 0 failed, 18 ignored (`cpu/suite.log`);
- `cargo clippy --release -p memra-server --all-targets -- -D warnings`: clean; `cargo fmt --all
  -- --check`: clean.

Red arms (`harness/red_arms.py`): each changes one thing, the fix is restored after each, and the
named test fails with cargo exit 101 and its own panic (`cpu/red/<arm>.log`).

| arm | what it changes | the test that fails, and how |
|---|---|---|
| `ladder-reap-off` | `reclaim_parked_ladder` does not reap (main's program for the ladder) | `the_parked_ladder_stops_after_the_one_eviction_its_reap_funds`: `on_demand=true: one eviction funds the arrival`, `left: 3, right: 1` |
| `climb-reap-off` | `reclaim_climb` does not reap (main's program for the climb) | `the_restorable_climb_stops_after_the_one_eviction_its_reap_funds`: `Err("the parked-session ladder ran out before the longer entry's bytes covered the rest")` against `Ok(239000000)` |
| `retry-reap-off` | `alloc_with_single_reclaim_retry` does not reap (main's program for the five retries) | `a_reclaim_retry_gets_the_evicted_bytes_after_its_reap`: `Err("DriverError(CUDA_ERROR_OUT_OF_MEMORY, out of memory)")` against `Ok(())` |
| `main-worker` | main's `worker.rs` with the census module alone appended | the census: `lost: reclaim_parked_ladder( ...`, `left: 0, right: 1` |
| `ladder-site-off` | the reclaim-on-defer ladder is handed a reap that does nothing | the census |
| `drain-reap-off` | the admission drain trims without its reap | the census |
| `settle-reap-off` | the prefix reclaim's settle does not reap | the census |
| `spec-miss-reap-off` | the spec pool's miss retries without its reap | the census |

Round 0 (`cpu/r0/`): the same checks on the first cut, whose census wrapped two single needles in
`concat!`. Its tests and suite passed, clippy refused the two `concat!` (`useless_concat`), and
the run was stopped after the first red arm (red as above) while the second compiled; the head
replaces the two `concat!` with plain literals and nothing else.

## GPU cells

None ran, and none is owed. The change does work only where the on-demand KV planes are armed:
the full-power RTX PRO 6000 Blackwell Workstation Edition
(`HardwareTarget::is_pro6000_workstation`: the name holds `Workstation` and not `Max-Q`). The
cards memra's cell evidence and daily regression run on (the RTX 5090 Laptop GPU and the RTX PRO
6000 Blackwell Max-Q Workstation Edition for sm_120a, GB10 for sm_121a) all run pooled planes
(`[kv-vmm] door=OFF`), where every new reap returns at once and the extracted ladder is main's
loop. A serving cell on any of them would rerun main's program, and the landing's push evidence
(sm_120a and sm_121a: battery, serve-smoke, perf, `tools/cell_evidence.py need`) covers that.
Showing the defect itself needs a full-power Workstation Edition card; none was in this lane's
reach. The fake device stands in for the one hardware
fact the fix relies on, that a reap returns the graves' bytes to the driver's reading at once:
DAY37's A2 grow cell (`research/spill-b-20260919/DAY37.md`, GROW-G1) read driver free back to
the byte it read before the cache was built once the dropped cache's graveyard reaped.

## Left open

- **Admission's first reading.** Each worker loop runs admission before its tick-top reap, so the
  first headroom reading of a pass leaves out what the previous tick released and nobody has
  reaped yet: the graves of sessions retired or dropped by the park cap, and the trimmed tails of
  freshly parked sessions. When that reading is short, the reclaim sheds kv-flex and evicts prefix
  entries before the first reap (the settle's, now). Candidates: reap and read again before the
  reclaim starts when anything is pending (`vmm_pending_bytes`), or run the tick-top reap before
  admission. Not changed here: it moves where admission reads, inside the block memra#1258 and
  memra#1304 are changing.
- **The drain's reap covers the graveyard only.** Parked sessions' trimmed tails are reaped by the
  tick-top reap (`vmm_reap_tick`), not by `vmm_reap_for`; the drained re-read still leaves them
  out. An eviction is unaffected: a dropped plane goes to the graveyard whole, tails included.
- **kv-flex's shed** (`MEMRA_KV_FLEX`, default off) re-reads after evicting prefix entries with
  neither the settle nor a reap.
- **No hardware reading.** The defect and the fix run only on the full-power Workstation Edition
  (see GPU cells).
