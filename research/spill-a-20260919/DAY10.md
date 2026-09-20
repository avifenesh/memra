# WP-A day 10: the storage `Busy` flake (lead ruling 8)

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`, source `ce423a158` (lane tip
`b3dc864ce` merged with `origin/main` `30e9c7a38`). Task: `crates/memra-tier/tests/storage/day4.rs`
panics with `called Result::unwrap() on an Err value: Busy` at :143, :240, :309, :471 on this rig
(lane B: 3 of 53 on a quiet release rerun; 276/276 on the PRO card box). Decide race versus rig,
fix or label, no engine change without a receipt. Raw receipts: `day10-flake/`.

## Verdict

**Race, not rig.** A test-harness race between sibling tests in one process: four tests spawn a
child (`Command::spawn` / `status`), and between `clone3(CLONE_VM|CLONE_VFORK)` and the child's
`execve` the child holds a copy of the parent's whole descriptor table. Every `flock` the process
holds stays held for that window even after its owner closed the descriptor. A sibling test that
closes a lock descriptor and re-locks the same file on a fresh descriptor (`CatalogStore::open`
on `.catalog-owner`, `FileBackend::evict` through `lock_gc_gate` on `.ownership-gc`) gets
`EAGAIN`, which the engine reports as `Busy`. Class (b) of the ruling: the tests assumed that
closing the descriptor releases the OS lock at once, which holds only when no fork is in flight;
the completion signal they never waited on is "no sibling spawn in flight".

Not (a): the engine answers `Busy` because another process (the forked, not yet exec'd child)
really references the locked description; no ordering in `object_store`, `catalog` or the frozen
`conformance` schedules is wrong, and none was changed. Not (c): the failure is a `flock`
`EAGAIN`, not an open error; the 10/10 control ran on the same tmpfs `/tmp`; `O_DIRECT` opens on
this tmpfs succeed (kernel `7.0.0-31-generic`, probe in the session log); and the same :471
failure sits in lane B's rented-5090 log
(`research/spill-b-20260919/rented-5090-20260919/day9-vmm-build/tests.log:296`, release build,
default threads), so "this rig only" is refuted. 276/276 on the PRO box is one sample of a
probabilistic race, not a rig property.

## Evidence

All runs under `systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G`, debug profile,
`cargo test -p memra-tier --test storage --offline`, `TMPDIR` unset (`/tmp`, tmpfs), 24 CPUs,
load average 1.4 to 1.9 at every start (`*/run-N.start`), co-resident processes in
`*/run-N.procs` (an unrelated `python3` at ~98% of one core through the whole session, several
`claude`/`electron` processes). Runner: `day10-flake/run-battery.sh`. Binary before the fix:
`storage-1825ca21bf643b79`, sha256 prefix `93590fc4bf5cb403`.

### Before the fix (source `ce423a158`)

| battery | mode | runs | verdicts |
|---|---|---|---|
| `threads1` | `--test-threads=1` | 5 | 5 x `test result: ok. 53 passed; 0 failed` |
| `default` | default threads | 5 | 4 x `ok. 53 passed`; run-4 `test result: FAILED. 51 passed; 2 failed` (:209, :309) |
| `control-skip-spawn` | default, the 4 spawning tests `--skip`ped | 10 | 10 x `test result: ok. 49 passed; 0 failed; ... 4 filtered out` |
| `stress-default` | default threads | 10 | 8 x `ok. 53 passed`; run-7 `FAILED. 52 passed; 1 failed` (:309); run-10 `FAILED. 51 passed; 2 failed` (:209, :309) |
| `nocapture-failing-only` | default, `--nocapture`, only the 5 tests that have failed | 1 | `test result: ok. 5 passed; 0 failed; ... 48 filtered out` |
| `strace/attempt-1` | default, `--nocapture`, `RUST_BACKTRACE=1`, under `strace -f -tt -y` | 1 | `test result: FAILED. 52 passed; 1 failed` (:240), captured on the first attempt |

Failing sites seen here: :209 (`legacy_gc_dedup_other_store_and_live_lease_fences`,
`store.evict` after `drop(other)`), :240 (`legacy_gc_replays_crash_after_tombstone_and_before_unlink`,
`store.evict` after reopen), :309 (`review_catalog_recovers_pending_with_and_without_published_root`,
`CatalogStore::open` after `drop(store)`). Lane B's :143 (`CatalogStore::open` after `drop`) and
:471 (`store.evict` after a `Corrupt` evict dropped its gate) are the same shape: a fresh
descriptor re-locking a file whose previous descriptor was just closed. Every panic is
`called Result::unwrap() on an Err value: Busy`. The failing tests alone pass; removing only the
spawning tests removes every failure; one thread removes every failure.

### The syscall origin (`day10-flake/strace/attempt-1.strace`, one process, pid 2436064)

Thread 2436103 runs `legacy_gc_replays_crash_after_tombstone_and_before_unlink`; thread 2436108
runs `review_gc_live_cross_process_transaction_commits_byte_exact`; 2436112 is its child.

```
2436103 01:03:45.684659 flock(23</tmp/memra-spill-a-test-2436064-15/.ownership-gc>, LOCK_EX|LOCK_NB) = 0     reopen takes the GC gate
2436103 01:03:45.684710 flock(26</tmp/memra-spill-a-test-2436064-15/.ownership>, LOCK_SH) = 0                reopen takes shared lifetime
2436108 01:03:45.684746 clone3({flags=CLONE_VM|CLONE_VFORK|CLONE_CLEAR_SIGHAND, ...} <unfinished ...>       sibling spawn begins: fd table copied
2436103 01:03:45.684756 close(23</tmp/memra-spill-a-test-2436064-15/.ownership-gc>) = 0                      reopen drops its gate; the child's copy keeps the lock
2436112 01:03:45.684888 execve(".../storage-1825ca21bf643b79", [..., "--exact", "day4::review_gc_live_cross_proce"...] <unfinished ...>
2436103 01:03:45.684917 flock(31</tmp/memra-spill-a-test-2436064-15/.ownership-gc>, LOCK_EX|LOCK_NB <unfinished ...>  evict re-locks the gate on a fresh descriptor
2436103 01:03:45.684933 <... flock resumed>) = -1 EAGAIN (Resource temporarily unavailable)                    Busy, panics at day4.rs:240:25
2436108 01:03:45.684976 <... clone3 resumed>) = 2436112                                                       spawn returns only after exec
```

The backtrace in `strace/attempt-1.log` ends at `day4.rs:240:25` (`store.evict(&key()).unwrap()`).
The 17 `execve` PATH probes for `python3` in the strace have their directory component replaced
by `<path-entry>` (they name tool install directories on the rig and carry no evidence); nothing
else in the file was edited. strace changes timing, so its run is a mechanism receipt, not a rate.

### After the fix (same source plus the harness change below)

| battery | mode | runs | verdicts |
|---|---|---|---|
| `after-default` | default threads | 5 | 5 x `test result: ok. 53 passed; 0 failed` |
| `after-threads1` | `--test-threads=1` | 5 | 5 x `test result: ok. 53 passed; 0 failed` |
| `after-stress-default` | default threads | 20 | 20 x `test result: ok. 53 passed; 0 failed` |
| `final-default` | default threads, committed source | 5 | 5 x `test result: ok. 53 passed; 0 failed` |

Before: 3 of 15 default-thread runs failed (4 test instances of 795). After: 0 of 30 default-thread
runs failed (0 of 1590). Wall time per default run moved from about 1.05 s to about 1.65 s because
the four spawning tests now run without directory-owning siblings; the `--test-threads=1` time
moved from 1.68 s to 1.91 s under the same co-tenant.

## The change (test harness only, no engine file)

`crates/memra-tier/tests/storage/mod.rs`: a process-wide `RwLock` fence. `OwnedDirectory::new()`
holds it shared for the directory's lifetime (re-entrant per thread, so a test that owns several
directories takes one read guard); `OwnedDirectory::spawning()` and `Fence::exclusive()` hold it
exclusively for the whole test body. Every store a test opens is declared after its directory, so
it drops before the fence releases. `day4.rs`: the three child-spawning tests use
`OwnedDirectory::spawning()`. `telemetry.rs`: the `python3` spawn takes `Fence::exclusive()`.
No sleep, no retry, no relaxed assertion, no change to `object_store`, `catalog`, `io` or the
frozen `memra_tier::conformance` schedules, no new dependency.

Note for the lead, not a change: the engine contract is untouched and correct (`Busy` means
another open description holds the lock), but any production process that spawns children while
another thread re-locks a store file on a fresh descriptor sees the same transient `Busy`, because
the child's descriptor-table copy holds the lock until its `execve`. If that matters for serving,
it is a separate, receipted decision.

## CPU gate (all under the quota scope, logs in `day10-flake/gates/`)

Runner `day10-flake/run-gates.sh` (a first attempt through a zsh variable never ran the commands,
exit 127; its logs were overwritten by the real run).

| gate | exit | verbatim tail |
|---|---|---|
| `cargo fmt --all -- --check` | 0 | (no output) |
| `cargo test -p memra-tier -p memra-kv --offline` | 0 | 10 suites, `passed` sums to 265, `failed` to 0 |
| `cargo clippy -p memra-tier -p memra-kv --offline --all-targets -- -D warnings` | 0 | ``Finished `dev` profile [unoptimized + debuginfo] target(s) in 3.14s`` |
| `DOCS_RS=1 cargo clippy -p memra-engine --offline --all-targets -- -D warnings` | 0 | ``Finished `dev` profile [unoptimized + debuginfo] target(s) in 27.33s`` |
| `bash tools/check-flags.sh` | 0 | `check-flags: runtime literal reads=864`, `check-flags: no uncovered runtime names` |
| `git diff --check` | 0 | (no output) |

## Push

`git push origin lane/spill-a-20260919` after the merge was refused by the pre-push perf-ci gate:
`pre-push: engine files touched after the last perf-ci battery` listing nine `crates/memra-engine`
files that arrive from `origin/main` through the merge (`STATE.md` has the SHAs). No
`--no-verify`, no skip variable; the lead pushes.
