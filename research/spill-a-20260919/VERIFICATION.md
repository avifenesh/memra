# WP-A day-1 verification receipt — 2026-09-19

Repo `avifenesh/memra`; branch `lane/spill-a-20260919`;
source commit `44fbe0515cdeae6199db80e9369c254c45498c61`.
Base `c5a33b14ff7b6c75a3cd808dbc0ee4f10aa8b33f`.
**Local CPU milestone only. Not merged, pushed, deployed, GPU-qualified or
live-verified.** Contracts proposed; integration and real hardware gates pending.

## Executed checks and exact output

All commands ran inside `../wt-spill-a`. Raw command output is retained alongside
this receipt, including the failed full-engine check. Final source was unchanged
between testing and source commit; this receipt-only commit adds no Rust changes.

`cargo test --manifest-path research/spill-a-20260919/scaffold/Cargo.toml
-p memra-tier-spill-a-scaffold --offline` — exit 0. Full output:
`raw/test-final.log.gz`. Nonzero test target:

```text
running 15 tests
test exact_pread_eintr_partial_eof_overflow ... ok
test jsonl_escapes_strings_and_unknown_times_are_null ... ok
test fake_pool_reserves_demand_headroom_and_charges_padding ... ok
test retirement::cancelled_transfer_needs_disk_dma_and_consumer_retirement ... ok
test corruption_header_payload_padding_truncation_and_future_version_refused ... ok
test retirement::lost_transfer_completion_is_quarantined_on_drop ... ok
test bounded_reader_partial_acceptance_and_short_read_visibility ... ok
test unknown_completion_quarantines_not_reuses ... ok
test cancel_late_completion_keeps_source_and_slot_until_io_retires ... ok
test incomplete_enospc_short_write_and_interrupted_commit_never_publish ... ok
test corrupt_or_missing_chunk_blocks_commit_and_read ... ok
test root_collision_identity_and_immutable_conflicts_fail_closed ... ok
test real_filesystem_reopen_and_conflict_do_not_clobber ... ok
test publication_is_last_and_streams_object_larger_than_pool ... ok
test header_valid_vs_padded_and_full_integrity ... ok

test result: ok. 15 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.17s
```

Test logs are losslessly gzip-archived to preserve Cargo's exact trailing blank
lines without violating the repository whitespace gate (`gzip -cd <path>`).

Library, binary and doc-test harnesses each report zero tests; these are not added
to the 15-test count. Earlier 13-test result is `raw/test-initial.log.gz`, retained
rather than overwritten after B's retirement requirements added two tests.

`cargo check --manifest-path research/spill-a-20260919/scaffold/Cargo.toml
-p memra-tier-spill-a-scaffold --all-targets --offline` — exit 0, full output
`raw/check-final.log`; final line:

```text
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 2.33s
```

Repeated after source commit, exit 0:

```text
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.04s
```

- `cargo fmt --all -- --check` — exit 0, **empty stdout/stderr**;
  `raw/fmt-workspace.log` is empty.
- `cargo fmt --manifest-path research/spill-a-20260919/scaffold/Cargo.toml -- --check`
  — exit 0, **empty stdout/stderr**; `raw/fmt-scaffold.log` is empty.
- `git diff --check`, `git diff --cached --check`, and
  `git diff --check c5a33b14ff7b6c75a3cd808dbc0ee4f10aa8b33f HEAD`
  — exit 0, **empty stdout/stderr**. Staged receipt check repeats before commit.
- `tools/check-flags.sh` — exit 0, `raw/check-flags.log`:

```text
check-flags: runtime literal reads=864
check-flags: no uncovered runtime names
check-flags: every runtime MEMRA_* name resolves against 'docs/FLAGS.md' (no grandfather list)
```

`cargo check -p memra-engine --offline` — **exit 101**. Full output retained in
`raw/check-engine.log`. Build did run, but no engine compilation pass is claimed:

```text
  thread 'main' (14713998) panicked at crates/memra-engine/build.rs:314:62:
  spawn nvcc: Os { code: 2, kind: NotFound, message: "No such file or directory" }
```

Also warned no runnable nvcc/no GPU and emitted an existing unused AsRawFd import
warning in memra-gguf. The new CLI is separately compiled by the CPU scaffold;
lead must add workspace/engine dependency wiring before integrated engine tests.

## CLI / fixture checks

Executed actual compiled `scaffold/target/debug/storage-bench cpu-fixture
research/spill-a-20260919/cpu-smoke-owned 264` once. Python independently
validated the parsed JSON schema, exact SHA256 against `fixtures.json`, valid vs
padded byte counts and submitted I/O accounting. Exact validation output:

```text
CLI_SMOKE_PASS: schema=1 valid=264 padded=4096 io=57344 SHA256=fixture-pin; physical/RSS/GPU times=null
CLI_RED_PASS: existing-directory rc=1; unsupported-mode rc=1
```

`raw/cli-smoke.jsonl` is the actual single-run CPU output, not a scored bandwidth
row. `raw/cli-smoke.stderr.log` is empty. Reusing the nonempty directory and
requesting unsupported `roundtrip` both returned 1, recorded in `raw/cli-red.log`.
The task-owned CLI directory and all test temporary directories were removed.
Fixture and implementation SHA256 pins are in `source-manifest.json`.

## Unexecuted checks / blockers

- No GPU roundtrip, real pinned allocation, O_DIRECT/io_uring/GDS, NVMe physical
  throughput, model consumption, serving, PRO or four-card battery ran.
- Both allowed bounded SSH attempts returned 255 with
  `Connection closed by UNKNOWN port 65535`; no remote inventory ran.
- No transfer performance/default decision and no perf-board movement.
- Shared/frozen contracts, device ownership, B's permits/charged object leases,
  and D's peer backend remain lead decisions; see CONTRACTS-PROPOSAL.md.
- Receipt/mesh/LSP tools were not exposed to this worker; direct cargo checks and
  committed raw logs are the evidence. No environment/auth files were read.

## Ownership and cost

Existing spill_pread/pinned_host/model/KV/server/peer sources are unchanged. Only
A-reserved module/test/bin paths and A receipt namespace changed. Shared manifest,
contracts/lib and docs are untouched; patch fragments are in LEAD-FRAGMENTS.md.
Main's unrelated state was not absorbed. Worktree remains open for the agreed
multi-round engagement; it is not a closed lane to delete yet.

Approximately **0.4 agent-hours** this round against WP-A's **7 agent-day** budget;
GPU time zero. This is elapsed worker time, not an estimate that the rest of WP-A
or qualification is complete.
