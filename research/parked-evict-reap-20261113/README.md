# Every eviction a reading or a retry counts on reaps the graveyard first (memra#1278)

Verdict: fixed (memra#1278), on CPU evidence only; no card the push evidence runs on reaches the
code it changes. Under
on-demand KV planes, a dropped parked session's planes wait in the graveyard until a reap, and the
free-memory reading admission and its reclaims take (driver free plus the pool's cached blocks)
leaves the graveyard out. The reclaim-on-defer's parked ladder re-read that reading after each drop
without a reap: the evicted bytes never showed, and one arrival could drop every parked session.
memra#1268 fixed the pre-cohort reclaim's copy of the pattern. Main had nine more eviction sites
whose bytes a reading or an allocation retry then counted on, and two re-reads after drops (the
admission drain and the prefix reclaim's settle); each now reaps (`vmm_reap_for`) before the
reading or retry. On a fake device whose freed bytes appear only after a reap, the ladder, the
restorable spare's climb and the one reclaim retry stop after the one eviction that funds the
arrival. With main's program (the reap removed) the ladder spent all three parked sessions
(`left: 3, right: 1`), the climb ran out of parked sessions and dropped the restorable entry, and
the retry failed like the first allocation.

The independent review of the first cut (memra#1310) found the same defect where no eviction
call names it, and one more of the same family (memra#1314). Folded: an allocation retries once
when its reclaim evicted something or its reap released bytes, so a session an earlier arm of the
same `admit` dropped funds the retry; the failed prefix materialization's flush, the park-compact
and plain-affinity failures and each failed right-size rung reap before the next allocation; and a
path that allocates next trims the device pools after its reap, because an on-demand plane maps
from the driver and the pool keeps the blocks a flushed prefix entry freed. Every red arm fails
its named test; the memra-server suite, clippy and rustfmt pass on the head.

## The instances

Each row evicts or drops, then reads free memory or allocates what needs those bytes. Found on
main `df1d23a764` (the first cut) and by the review; the head is rebased on main `bbc6f83aea`.

| site | what it evicts or drops | what counts on the bytes | the reap now |
|---|---|---|---|
| reclaim-on-defer parked ladder (admission) | globally oldest parked session that frees its planes, one at a time | the headroom re-read after each drop (the issue's line) | `reclaim_parked_ladder` reaps between the eviction and the read (`"reclaim-on-defer"`) |
| restorable spare's climb (admission) | the same, one at a time | the unbooked shortfall re-read after each drop | `reclaim_climb` reaps between them (`"reclaim-on-defer climb"`) |
| park-compact grow, plain-affinity grow, cold cache allocation (`admit`) | all unleased prefix entries and the oldest parked session | the one cache-allocation retry | `alloc_with_single_reclaim_retry` runs `vmm_retry_reap` (reap, then trim) before its retry |
| speculative grid rewind and affinity grow (`spec_rewind_oom_reclaim` through `spec_lend_rewind`) | the same | the one retry | the same |
| spec pool, learned evict-first (`admit`) | the request's whole spec pool | the session allocation right after | `"spec pool evict-first"`, reap then trim |
| spec pool miss (`admit`) | the request's whole spec pool | the one retry | `vmm_retry_reap` (`"spec pool miss"`), on the retry rule below |
| admission drain (idle reject) | nothing itself: the graves of the last tick's retires and evictions, which the tick-top reap releases only after admission | the drained re-read after the fence and trim | between the fence and the trim (`"admission-drain"`), as on every reclaim path |
| prefix reclaim settle (`settle_reclaimed_prefix_bytes`) | prefix entries; one that held the last lend of an adopted plane (memra#1221 part 2) sends that plane to the graveyard | the re-read after the settle | after the settle's fence; the receipt prints `graveyard_reaped_bytes` |
| **review:** failed prefix materialization (`admit`) | the restore's partly mapped planes; on an allocation failure every unleased prefix entry too | the cold cache allocation | `flush_then_reap`: flush, reap, trim (`"prefix materialization flush"`); a restore that failed otherwise reaps and trims too |
| **review:** any session an earlier arm of the same `admit` dropped, and a failed attempt's own planes | no eviction call: a drop | the allocation's one retry | the retry rule: reclaim, reap, and retry when the reclaim evicted something **or the reap released bytes** |
| **review:** park-compact grow failure, plain-affinity resume failure (`admit`) | the parked session (and a grown cache its restore failed into) | the cold cache allocation | after the drop (`"park-compact grow failed"`, `"plain-affinity resume failed"`); the plain-affinity arm drops its entry before the reap, not at the end of its block |
| **review:** right-size ladder (`spec_right_size_ladder`) | each failed rung's partly mapped planes, or the session a headroom-less landing drops | the next rung, and the tokenwise path's allocation after the last | before the first rung and after each failed one (`"spec right-size"`) |
| **review:** every reclaim retry above (memra#1314) | prefix entries: their blocks go to the device pool, whose release threshold is pinned | the retry's on-demand maps, which take memory from the driver alone | `vmm_reap_trim_for`: the reap, then the pool trim (the on-demand grow reclaim's order) |
| **review:** kv-flex shed (`MEMRA_KV_FLEX`, default off) | prefix entries above the floor | the admission reading after it | the shed reaps what it shed (`"kv-flex shed"`) |

Already reaping before this change: the oom-teardown, admin-trim and batch-oom teardowns (DAY37
addendum D, day 49 addendum C), the on-demand grow's reclaim retry, the admission ensure retry and
the pre-cohort reclaim (memra#1266 review 2). The park cap (`prepare_park`) evicts by count and
reads nothing; the prefix budget's hold and the peer probe's degrade flush and read nothing.

Pooled planes, which is every card but the full-power RTX PRO 6000 Blackwell Workstation Edition:
`vmm_reap_for` returns 0 on its first line (`kv_vmm::armed()`, a `OnceLock` read),
`vmm_reap_trim_for` and `vmm_retry_reap` return before the reap and the trim, and nothing was added
to a path where the first allocation succeeds. A retry there is the reclaim's alone, as on main.

The reap's line: a reap after an eviction that released nothing and left nothing pending prints
nothing. The three teardowns keep their line on every call, because the day-49 readers count one
`[kv-vmm] reap (batch-oom)` line per teardown and the 15 tracked ones read `released=0`
(`research/spill-b-20260919/day49d-read.py`).

## Tests (`crates/memra-server/src/worker.rs`, `parked_evict_reap_1278_tests`)

The fake device: dropping a parked session sends its bytes to the graveyard under on-demand
planes and to the pool's cached blocks under pooled planes; a flushed prefix entry's blocks are
the pool's under either. The reading is driver free plus pool cached. A reap hands the graveyard
to the driver, a trim the pool's cached blocks. A session cache maps from the driver alone under
on-demand planes, and a failed build leaves what it mapped in the graveyard; under pooled planes
it takes the pool's cached blocks first.

- `the_parked_ladder_stops_after_the_one_eviction_its_reap_funds`: 10 GiB free, three parked
  sessions of 4 GiB, an arrival that needs 12 GiB. One eviction, two sessions stay, the order is
  read, evict, reap, read; the same under pooled planes (the reap does nothing there). An arrival
  nothing covers still takes all three; a lost reading stops the ladder on its last reading.
- `the_restorable_climb_stops_after_the_one_eviction_its_reap_funds`: short by 259 MB with a
  249 MB longer entry and 20 MB parked sessions; one eviction leaves 239 MB, and the longer entry
  goes.
- `a_reclaim_retry_gets_the_evicted_bytes_after_its_reap`: 2 GiB free, one 4 GiB parked session,
  a 5 GiB allocation: the retry succeeds.
- `a_reclaim_retry_runs_on_the_bytes_its_reap_alone_released` (review): 2 GiB free, nothing to
  evict, a 4 GiB grave an earlier arm left, a 5 GiB allocation: the reap releases the grave and
  the retry succeeds. Nothing evicted and nothing released: no retry. Pooled planes: the first
  attempt takes the pool's blocks and nothing reaps.
- `a_reclaim_retry_maps_the_flushed_prefix_blocks_after_its_trim` (review, memra#1314): 2 GiB
  free, 4 GiB of prefix entries, a 5 GiB allocation: order alloc, flush, reap, trim, alloc, and
  the retry succeeds. Pooled planes: alloc, flush, alloc, no reap and no trim.
- `a_failed_materialization_flush_is_reaped_before_the_cold_allocation` (review): 3 GiB free,
  4 GiB of prefix entries, a 6 GiB session cache, with one 4 GiB parked session and with none.
  The restore's cache fails and leaves 3 GiB in the graveyard; after the flush, the reap and the
  trim the cold allocation fits on its first attempt and the parked session stays.
- `a_failed_right_size_rung_is_reaped_before_the_next_one` (review): 10 GiB free, a ladder from
  16 GiB to 4 GiB: the 16 GiB rung fails, the 8 GiB rung lands. Nothing fitting down to the need
  tries each rung once and reaps each, the last too.
- `a_reap_that_released_nothing_prints_no_line_outside_the_teardowns` (review): the reap's line
  and the settle receipt's `graveyard_reaped_bytes` note.
- `every_eviction_a_reading_or_a_retry_counts_on_reaps_first` (the census): 63 needles over the
  live code with comments cut. Every call of an eviction entry point is counted (2
  `evict_oldest_parked_freeing(`, 7 `evict_oldest_parked(`, 11 `px.evict_all()`, 7
  `evict_to_bytes(`, 2 `evict_all_demoting(`, 1 `.evict_entry(`, 2 `.shed(`, the spec pool's 2
  clears, the 3 whole-pool teardowns) and each is named with the reap that follows it; so are
  the retry rule, the reap helpers' bodies, the drops reviewed in `admit`, and the three kinds of
  reap call (8 `vmm_reap_for("`, 7 `vmm_reap_trim_for(`, 6 `vmm_retry_reap(`). A new parked
  eviction or prefix flush fails a count. The census cannot see a drop: a session or cache that
  goes out of scope or is set to `None` has no call to count. For those the retry rule is the
  net at an allocation; a drop a free-memory reading follows has none. The memra#1266 census
  counts 8 `vmm_reap_for("` sites and pins the reap-then-trim order.

Changed existing tests: `cache_alloc_oom_reclaim_retries_exactly_once` (the reap runs after every
first failure; a third case retries on the reap alone) and the lend-refused rewind's closure.

## CPU evidence (`cpu/`)

On the CPU build host (`harness/dd_check.sh`, cores 14 to 31), `MEMRA_CUDA_ARCH=120a`, on
`a4273d7f98`, the fold's commit on main `bbc6f83aea` (`worker.rs` sha256 `65efc1980c908fdf...`,
`cpu/summary.txt`). Green in the release profile:

- the nine lane tests: 9 passed; the touched tests (the memra#1266 census, the one-retry bound,
  the lend-refused rewind, the lender-skipping ladder, the pre-cohort reclaim's 7, the restorable
  spare's 4, kv-flex's 7): all passed;
- memra-server library suite: 1,272 passed, 0 failed, 18 ignored (`cpu/suite.log`);
- `cargo clippy --release -p memra-server --all-targets -- -D warnings`: clean; `rustfmt
  --edition 2024 --check` on `worker.rs`: clean.

In the checkout (`cpu/checkout-checks.log`; these read git): the doc links, the public boundary,
the support-records contract with its unit tests, and `git diff --check` against main. The logs
are the tools' output with the lane's scratch path written `<lane>` and the blank lines at the
end of each file removed (what `git diff --check` refuses).

Red arms (`harness/red_arms.py`): each changes one thing, the fix is restored after each (the
restored file compares equal), and the named test fails with cargo exit 101 and its own panic
(`cpu/red/<arm>.log`; the dev profile, rebuilt incrementally; `cpu/red/green-dev.log` is the fix
in that profile, 9 passed). `cpu/summary.txt` has one line per arm with the test and the panic
text it was checked for.

| arm | what it changes | the test that fails, and how |
|---|---|---|
| `ladder-reap-off` | `reclaim_parked_ladder` does not reap (main's program for the ladder) | `the_parked_ladder_stops_after_the_one_eviction_its_reap_funds`: `on_demand=true: one eviction funds the arrival`, `left: 3, right: 1` |
| `climb-reap-off` | `reclaim_climb` does not reap (main's program for the climb) | `the_restorable_climb_stops_after_the_one_eviction_its_reap_funds`: `Err("the parked-session ladder ran out before the longer entry's bytes covered the rest")` against `Ok(239000000)` |
| `retry-reap-off` | `alloc_with_single_reclaim_retry` does not reap (main's program for the retries) | `a_reclaim_retry_gets_the_evicted_bytes_after_its_reap`: `Err("DriverError(CUDA_ERROR_OUT_OF_MEMORY, out of memory)")` against `Ok(())` |
| `retry-rule-off` | the retry runs only when the reclaim freed something (the first cut's rule) | `a_reclaim_retry_runs_on_the_bytes_its_reap_alone_released`: the same `Err` against `Ok(())` |
| `retry-trim-off` | `reap_then_trim` does not trim (the first cut's retries, memra#1314) | `a_reclaim_retry_maps_the_flushed_prefix_blocks_after_its_trim`: `on_demand=true`, the same `Err` against `Ok(())` |
| `flush-reap-off` | `flush_then_reap` does not reap (the first cut's materialization flush) | `a_failed_materialization_flush_is_reaped_before_the_cold_allocation`: `the cold allocation evicted no parked session`, `left: 0, right: 1` |
| `rung-reap-off` | `spec_right_size_ladder` does not reap a failed rung (the first cut's ladder) | `a_failed_right_size_rung_is_reaped_before_the_next_one`: `on_demand=true: the second rung lands`, `None` against `Some(((), 8589934592))` |
| `reap-line-always` | `vmm_reap_line` prints for every reap | `a_reap_that_released_nothing_prints_no_line_outside_the_teardowns`: `Some("[kv-vmm] reap (reclaim-on-defer) released=0 pending_graves=0")` against `None` |
| `main-worker` | main's `worker.rs` (`bbc6f83aea`) with the census module alone appended | the census: `lost, moved or new: reclaim_parked_ladder( ...`, `left: 0, right: 1` |
| `ladder-site-off` | the reclaim-on-defer ladder is handed a reap that does nothing | the census, on that site's needle |
| `drain-reap-off` | the admission drain trims without its reap | the census |
| `settle-reap-off` | the prefix reclaim's settle does not reap | the census |
| `spec-miss-reap-off` | the spec pool's miss retries without its reap | the census |
| `flush-site-off` | the materialization flush is handed a reap that does nothing | the census |
| `park-compact-fail-off` | the park-compact grow's failure drops the session without a reap | the census |
| `affinity-fail-drop-off` | the plain-affinity failure reaps before its entry drops | the census |
| `right-size-site-off` | the right-size ladder is handed a reap that does nothing | the census |
| `shed-reap-off` | kv-flex's shed does not reap what it shed | the census |
| `probe-prefix-reread` | the review's probe: a new `px.evict_all()` then `admission_headroom`, no settle and no reap | the census: `lost, moved or new: px.evict_all()`, `left: 12, right: 11` |
| `probe-parked-reread` | a new `evict_oldest_parked_freeing` then `admission_headroom`, no reap | the census: `lost, moved or new: evict_oldest_parked_freeing(`, `left: 3, right: 2` |

Earlier rounds, kept as they ran. Round 1 (`cpu/r1/`): the first cut, main `df1d23a764` with
`6ebd6493ea`'s `worker.rs` (sha256 `587120d661c80bc4...`), with the harness of that commit: 4 lane
tests, suite 1,266 passed, 0 failed, 18 ignored, clippy and fmt clean, eight red arms. Round 0
(`cpu/r0/`): the same checks on a census that wrapped two single needles in `concat!`; clippy
refused them (`useless_concat`) and the run stopped after the first red arm.

## GPU cells

None ran, and none is owed. The change does work only where the on-demand KV planes are armed:
the full-power RTX PRO 6000 Blackwell Workstation Edition
(`HardwareTarget::is_pro6000_workstation`: the name holds `Workstation` and not `Max-Q`). The
cards memra's cell evidence and daily regression run on (the RTX 5090 Laptop GPU and the RTX PRO
6000 Blackwell Max-Q Workstation Edition for sm_120a, GB10 for sm_121a) all run pooled planes
(`[kv-vmm] door=OFF`), where every new reap and trim returns at once and the extracted ladders
are main's loops. A serving cell on any of them would rerun main's program, and the landing's push
evidence (sm_120a and sm_121a: battery, serve-smoke, perf, `tools/cell_evidence.py need`) covers
that. Showing the defect itself needs a full-power Workstation Edition card; none was in this
lane's reach. The fake device stands in for two hardware facts. A reap returns the graves' bytes
to the driver's reading at once: DAY37's A2 grow cell (`research/spill-b-20260919/DAY37.md`,
GROW-G1) read driver free back to the byte it read before the cache was built once the dropped
cache's graveyard reaped. A block the pool caches is not the driver's until a trim: the reclaim
settle's receipts (memra#346, memra#445: `effective free 69204MB -> 69204MB` after a 43.5 GB
eviction, until the settle's trim).

## Left open

- **Admission's first reading (memra#1313).** Each worker loop runs admission before its tick-top
  reap, so the first headroom reading of a pass leaves out what the previous tick released and
  nobody has reaped yet: the graves of sessions retired or dropped by the park cap, and the
  trimmed tails of freshly parked sessions. When that reading is short, the reclaim sheds kv-flex
  and evicts prefix entries before the first reap of the pass, or one parked session at most.
  Not changed here: it moves where admission reads, inside the block memra#1258 and memra#1304
  changed.
- **The drain's reap covers the graveyard only.** Parked sessions' trimmed tails are reaped by the
  tick-top reap (`vmm_reap_tick`), not by `vmm_reap_for`; the drained re-read still leaves them
  out. An eviction is unaffected: a dropped plane goes to the graveyard whole, tails included.
- **A drop a reading follows.** The census counts eviction calls. A session dropped by scope or
  assignment right before a free-memory reading, with no allocation retry behind it, would pass
  it. None is known.
- **kv-flex's shed** (`MEMRA_KV_FLEX`, default off) now reaps, but still re-reads without the
  settle's fence, so entry frees still in flight read as used.
- **The trim is the whole cache.** The paths that allocate next trim the pools to zero, the
  on-demand grow reclaim's form, where the settle keeps the blocks parked before the reclaim.
  They are failure paths; a kept-blocks form was not measured.
- **No hardware reading.** The defect and the fix, memra#1314's included, run only on the
  full-power Workstation Edition (see GPU cells).
