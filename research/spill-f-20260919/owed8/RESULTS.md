# OWED 8: per-stage spill counters (2026-09-25)

**Landed CPU-verified; the counters' GPU values are owed with B3.** No new env read. Registration:
`CPU-PREREG.md` OWED 8.

## What changed

- `PreadStats` (`crates/memra-engine/src/spill_pread.rs`, final SHA-256 `8a904895...`): `worker_read_ns`
  (positioned-read wall time on worker threads, carried back in each completion),
  `demand_read_ns` (blocking `pread` mode on the owner), `wait_ns` (owner blocked on a worker
  completion in `wait_worker` or on an H2D event in `wait_for_one`), `h2d_submits` (known and
  unknown-completion submissions), beside OWED 7's `overread_bytes`. The pool's drop line prints
  all of them.
- `memra_engine::SpillStageStats` and `Engine::moe_pread_stage_stats()`; the 7-field
  `moe_pread_stats()` tuple is unchanged, so no caller breaks.
- `run-gen` prints `spill stages DECODE-WINDOW: worker_read_ms= demand_read_ms= wait_ms= h2d_submits= overread_bytes=`
  after its existing `spill worker DECODE-WINDOW` line, and the `STEADY-STATE rep` twin.
- `memra-server`'s existing `MEMRA_SPILL_STATS=1` snapshot appends the same fields;
  `docs/FLAGS.md` `MEMRA_SPILL_STATS` row updated.
- Cost: two `Instant::now()` per read and per blocking wait, off the GPU stream.

## CPU gates (raw logs in `owed8/cpu/`)

| Gate | Result |
|---|---|
| `cargo test -p memra-engine --lib spill_pread` (adds `stage_stats_deltas_and_fields`) | `test result: ok. 10 passed; 0 failed; 2 ignored` (before and after `cargo fmt`) |
| `cargo clippy -p memra-engine --lib --tests --bin run-gen -- -D warnings` | exit 0 |
| `cargo clippy -p memra-server --lib -- -D warnings` | exit 0 |
| `cargo fmt --all -- --check`, `git diff --check`, `tools/check-flags.sh` | exit 0, clean, "no uncovered runtime names" |

The ignored CUDA test `direct_worker_overread_preserves_exact_bytes` now also asserts
`worker_read_ns > 0` and `h2d_submits == 0` for aborted reads; it runs with OWED 7's GPU gates.
