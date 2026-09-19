# PR #518 review fixes — CPU verification receipt

Repository: memra. Branch: `lane/spill-integ-20260919`. Date: 2026-09-19.
Baseline: `64bfc718f61266ac4893c5f6242a1a5c0a91983a` (clean on entry).
Integrated source checked: `95baf3e1c899406a26efdd3c2fec15b14dc0f6a8`. This receipt-only commit does not change code.
Scope: the three reviewed defects; no merge, GPU execution, native CUDA wiring,
or serving qualification is claimed. Wire v1 and frozen trait shapes are unchanged.

## Finding 0 — workspace dependency already corrected

Inspected `crates/memra-kv/Cargo.toml`: `memra-tier = { workspace = true }`.
No change needed.

## Finding 1 — disappearing served stamps permit starvation

Root cause: pruning removed idle tenants immediately. A returning tenant then scored
zero, ahead of a tenant whose second request remained queued. The regression queues
B1/B2/A3, serves B1/A3, releases their charges, then queues A4: B2 must precede A4.

Fix: protect queued and in-flight tenants; governor tracks each issued charge's tenant
until successful release. Retain at most the existing queue limit of **idle** stamps,
evicting oldest idle entries first. `evicted_floor = max(all evicted stamps)` replaces
zero for unknown history. This is the lead-approved fairness floor: even forgotten
recent tenants cannot jump ahead of older queued stamps. Priority/deadline/FIFO and
blocked-head rules remain unchanged. Scheduler admission and release use the same rule.
Active history is bounded by live requests/charges, not evicted to satisfy the idle cap.

Commit: `f05c646caee4db550bd9ad4b08a82f9fabcd21da`.
Tests:
- `integration::governor_rearrival_cannot_starve_backlogged_tenant`
- `tiered::tests::scheduler_rearrival_cannot_starve_backlogged_tenant`
- `tier::governor::tests::fairness_idle_cap_keeps_inflight_and_queued_stamps_and_eviction_floor`
- `tiered::tests::scheduler_eviction_floor_protects_backlog_under_priority_churn`

The latter two exercise protected in-flight/queued history, bounded idle retention,
oldest-first eviction and priority churn past the cap. The original fairness tests remain.

Before command (new regressions, unchanged baseline implementation):
`cargo test -p memra-tier -p memra-kv --offline rearrival_cannot_starve --no-fail-fast`

```text
integration::governor_rearrival_cannot_starve_backlogged_tenant ... FAILED
assertion `left == right` failed
  left: 4
 right: 2
tiered::tests::scheduler_rearrival_cannot_starve_backlogged_tenant ... FAILED
assertion failed: s.tick(...).contains(&(id, Ok(Progress::Phase(Phase::Reserved))))
error: 2 targets failed: memra-kv --lib; memra-tier --test bank
```

After (same command, integrated source):
```text
test tiered::tests::scheduler_rearrival_cannot_starve_backlogged_tenant ... ok
test integration::governor_rearrival_cannot_starve_backlogged_tenant ... ok
```

## Finding 2 — rounded final extents exceeded readable payload

Root cause: `plan_reads` rounded the last segment to granularity even when the object
exposed only its valid payload extent. 8191/8193-byte tensors could not stage tail rows.

Fix: first require the entire segment storage range to exist, then clamp **only the
coalescing extension** to the reader's extent. ObjectReader continues to expose the
logical tensor extent, rejects reads past it, and copies short tails from fully validated
chunk payloads. Existing initialized pool padding is not exposed as invented payload.
No padded-readable-extent fiction and no transfer-engine reporting change were introduced.

Commit: `e241033c5d65d7e3589359e5b708c415e2ed4008`.
Tests: `integration::object_reader_tail_rows_8191` and
`integration::object_reader_tail_rows_8193` use real FileBackend objects and 512-byte
chunks. Each stages the last byte, a 31-byte tail row and a 513-byte straddling row,
compares all output bytes, asserts every requested range is within the tensor extent,
checks exact final extent termination, rejects a one-byte overrun, drains all transfers,
and verifies quota returns to zero. Test directories are removed by a cleanup guard.
The existing planner test now accepts clipped coalescing padding but retains explicit
truncated-payload, invalid-granularity and overflow refusals.

Before command:
`cargo test -p memra-tier --offline object_reader_tail_rows --no-fail-fast`
```text
integration::object_reader_tail_rows_8191 ... FAILED
called `Result::unwrap()` on an `Err` value: InvalidLayout
integration::object_reader_tail_rows_8193 ... FAILED
called `Result::unwrap()` on an `Err` value: InvalidLayout
test result: FAILED. 0 passed; 2 failed
```
After (same command):
```text
test integration::object_reader_tail_rows_8191 ... ok
test integration::object_reader_tail_rows_8193 ... ok
```

## Finding 3 — completion confused logical payload with framed I/O

Root cause: `Completion::require` compared both logical valid bytes and physical I/O
bytes to caller expectations. Native CPU storage transfers report padded chunks plus
extent headers, not the hierarchy's guessed segment storage length.

Fix: compare exact `valid_bytes` only; `ShortIo` reports those logical lengths.
`io_bytes` remains physical submitted-byte telemetry. Keep SegmentExpectation's field
for API compatibility, document it as ignored, and use logical lengths in KV/bank
expectations rather than a misleading zero. Wire v1 is unchanged.

CPU transfer engines still report their true framed counts. KV and contracts NVMe
fakes now report padded chunk + header bytes; actual DMA fakes keep DMA byte counts.
Short-payload fault injection in KV/contracts/peer tests now mutates `valid_bytes`,
and the shared conformance check checks logical lengths instead of pinning telemetry.
No missing/short/corrupt/fence schedule was removed or weakened.

Bank host completions describe slot-to-record materialization (`storage_bytes` copied,
`valid_bytes` logical). Framed storage I/O belongs to ObjectReader's underlying transfer
completion and aggregate telemetry; assigning a coalesced physical read to every logical
record would double-count it. A code comment makes this distinction explicit.

Commit: `95baf3e1c899406a26efdd3c2fec15b14dc0f6a8`.
Test: `wire::completion_requires_logical_payload_not_framed_io` accepts framed I/O of
8192 bytes for a 3-byte logical payload; changing valid bytes to 2 returns exactly
`ShortIo { expected: 3, actual: 2 }`.

Before command:
`cargo test -p memra-tier --offline completion_requires_logical_payload_not_framed_io`
```text
wire::completion_requires_logical_payload_not_framed_io ... FAILED
called `Result::unwrap()` on an `Err` value: ShortIo { expected: 4, actual: 8192 }
```
After (same command):
```text
test wire::completion_requires_logical_payload_not_framed_io ... ok
```

## Integrated checks — actually run, all exit 0

| Command | Result |
|---|---|
| `cargo fmt --all -- --check` | PASS |
| `cargo check -p memra-tier -p memra-kv --offline --all-targets` | PASS, macOS host |
| Same check with `--target x86_64-unknown-linux-gnu` | PASS, Linux cross-compile only |
| `cargo test -p memra-tier -p memra-kv --offline --no-fail-fast` | PASS, 197 unit/integration + 4 compile-fail doctests |
| `cargo clippy -p memra-tier -p memra-kv --offline --all-targets --no-deps -- -D warnings` | PASS |
| `git diff --check` | PASS |
| `bash tools/check-flags.sh` | PASS, 864 literal reads, no uncovered names |

Full test counts (in command output order: KV, tier library, bank, contracts, peer,
placement, storage, KV docs, tier docs):
```text
test result: ok. 53 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 38 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.49s
test result: ok. 45 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.08s
test result: ok. 17 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 37 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.64s
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.17s
```

Known unrelated warning: macOS `memra-gguf/src/source.rs:20` has an unused
`std::os::fd::AsRawFd` import. It is an unchanged dependency warning; requested
no-deps clippy passes. Linux target check is clean. No GPU/nvcc available, so GPU,
serving and hardware qualification were not run. No performance/default claims.

Worktree was clean at the checked code commit. No unrelated changes were absorbed.
Push uses the configured `tools/hooks` pre-push hook; no bypass or merge authorized.
