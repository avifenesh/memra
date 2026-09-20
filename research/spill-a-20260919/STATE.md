# WP-A day 10 resumable state

- Lane `lane/spill-a-20260919`; Linux worktree `wt-spill-a`. Merge of `origin/main` `30e9c7a38`: `ce423a158`.
- Day 10 (lead ruling 8, storage `Busy` flake): RACE, not rig. Harness fix `5ac2952c1`; the receipts commit that adds this file is the branch tip.
- Mechanism: a sibling test's `Command::spawn` copies the fd table between `clone3(CLONE_VFORK)` and `execve`, so a just-closed `flock` stays held and a fresh `try_lock` reads `Busy`. strace receipt: `day10-flake/strace/attempt-1.strace`; write-up `DAY10.md`.
- Fix: process-wide RwLock fence in `tests/storage/mod.rs`; spawning tests hold it exclusively. No engine file, no sleep, no retry, no relaxed assertion, no frozen-schedule change.
- Before: default threads 3 of 15 runs failed (threads=1 5/5; spawn tests skipped 10/10). After: 0 of 30 default runs, 5/5 threads=1, final 5/5 on the committed source.
- Not rig-specific: the same :471 failure is in lane B's rented-5090 log; PRO 276/276 is one sample of a probabilistic race.
- CPU gate green under the quota: fmt, tier+kv tests (265 passed), clippy tier/kv and engine (`DOCS_RS=1`) `-D warnings`, check-flags (864), `diff --check`.
- Push refused by the pre-push perf-ci gate (nine `crates/memra-engine` files arriving from main through the merge). No `--no-verify`, no skip variable; the lead pushes.
- Lead note, no change made: a production process that spawns children while another thread re-locks a store file on a fresh descriptor sees the same transient `Busy`; a separate receipted decision if it matters for serving.
- Day 9 native state unchanged: `day9/RESULTS.md`, `V13-BINDING.md`; io_uring remains DEFERRED (`IO-BASELINE.md`).
