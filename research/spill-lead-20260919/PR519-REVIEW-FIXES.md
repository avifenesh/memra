# PR #519 — WP-A catalog / object-store review fixes

Repository: **avifenesh/memra**. Branch: **lane/spill-integ2-20260919**.
Base: `9612aa34e5dedf72a502abe20436248590d9fc47`.
Checked source: **`2dbe6f3ae37266673cf33b9ab496b110ea66a3c8`**.
This is a CPU/cross-compile review-fix milestone, not a main merge, release,
deployment, GPU qualification, or live-serving verification. The PR is not merged.

## Finding → cause → fix → regression → commit

| Finding | Root cause | Fix | Before / after regression | Commit |
|---|---|---|---|---|
| 1. Orphaned catalog `.pending` wedges key reuse | Publication links the pending inode to the root, but restart/install and collection never reconcile that staging link. Pre-publication orphan indices also refuse retries. | Under existing exclusive catalog ownership, remove an abandoned pending link. Preserve an existing immutable root and return Conflict; with no root/tomb, discard private partial indices and install anew. Tombstoned objects remain owned by collect, which removes pending before releasing the tomb. | `review_catalog_recovers_pending_with_and_without_published_root`: before, install returns Conflict for the unpublished crash fixture; after, both unpublished retry and published-root reconciliation pass, followed by tombstone/collect/reinstall without a stranded pending link. | `63cf6b714294c12997468dc1fd17f3aa3b47fea8` |
| 2. Shared → exclusive conversion has an unprotected window | Explicit unlock drops the lifetime lease fence before exclusive ownership. `flock` upgrades themselves are not portably atomic either. | A directory-local `.ownership-gc` gate serializes both admission and lock conversion. Acquire it before releasing shared ownership; retain it through exclusive GC and shared-lock restoration. Other owners still force Busy. If restoration fails, retain the gate until process exit rather than lose the fence. | `review_gc_upgrade_window_keeps_other_stores_lease_fenced`: two ExtentStores; the upgrading store has a live lease on K while collecting another key. A private synchronous hook attempts peer eviction of K precisely after unlock. Before: peer eviction **Ok**, violating the fence. After: **Busy**, new admission also Busy, and the original leased bytes still read exactly. | `2dbe6f3ae37266673cf33b9ab496b110ea66a3c8` |
| 3. GC ignores uncommitted transactions | Only published roots contribute references; a put sharing a victim's chunk loses its backing before commit. | Per-store transaction digest registry contributes protected references to the sealed eviction permit. FileBackend also retains txn-scoped chunk hardlinks and an independently locked owner file until commit/cancel. GC scans these references, and only proven-abandoned transactions are eligible for bounded cleanup. | `review_gc_preserves_uncommitted_transaction_chunks`: before, commit after eviction returns **NotFound**; after, commit/read are byte-exact. Additional tests exercise explicit cancel, corrupt staging as a no-op failure, live child-process commit, process exit without destructors, and bounded stale cleanup. | `2dbe6f3ae37266673cf33b9ab496b110ea66a3c8` |
| 4. Failed verification revokes visibility | Root → tomb rename precedes candidate decoding and the complete reference scan. | Decode the candidate and scan all other roots/staging metadata first. Rename only after the deletion plan is verified, then retain the durable tombstone until unlink/sync completes. | `review_gc_verification_failure_does_not_revoke_visibility`: an unrelated malformed root makes eviction fail. Before: lookup(K) is **None**. After: lookup still returns the original manifest, reads are byte-exact, and no tomb was created. | `372ea5c1753966f77d11aeff6df826369a5ddfe0` |

### Transaction ownership and bounded stale policy

- A transaction belongs to one ExtentStore; losing its caller handle does **not**
  retire its staged references. The registry/owner lock stays until explicit
  commit/cancel or destruction of the owning backend. A replacement store cannot
  resume a foreign transaction.
- `.txn-<pid>-<sequence>/chunk-<digest>` is a hardlink to immutable framed chunk
  bytes. PID is only a namespace component, never liveness evidence; create-new
  plus collision retry prevents PID reuse from overwriting old staging.
- Each transaction's `.owner` lock is held independently. Under exclusive store
  ownership, successful nonblocking acquisition proves the old owner is gone.
  A busy owner is always retained, with **no age-based timeout**. A missing owner
  file is the mkdir-before-owner crash case; exclusive store ownership proves no
  other writer can still be creating it. Other I/O errors fail closed.
- Each successful eviction reaps **at most 32** proven-stale directories; deferred
  directories continue to contribute references. This is an opportunistic bounded
  sweep during eviction, not a standalone background reaper or a time-to-reclaim
  guarantee. Stale-only stores await a subsequent eviction/recovery operation.
- Eligible stale chunks and the victim's chunks are removed only if absent from
  every committed root, live/deferred staging set, and local transaction registry.
  Unexpected staging names abort verification before any root rename or unlink.
- On commit, durable root publication precedes removal of staging links; on cancel,
  publication is explicitly revoked first. Cleanup failure retains retry ownership.
- Existing live-lease and unknown-completion fences remain in place. The new gate
  is a **store-directory lock**, not another GPU campaign lock. All concurrent
  writers/collectors must use this ownership protocol; mixed old/new binaries are
  not a qualified concurrency configuration.

### Failure boundaries

Verification errors before rename leave object visibility/backing unchanged.
An I/O failure **after** successful rename (unlink or sync failure) still retains
recoverable tombstone state; it is not reported as a completed eviction. The
existing tombstone-crash and unknown-shutdown regressions remain passing.

## Verification receipts

Gzip-preserved raw combined stdout/stderr: [`pr519-review/`](pr519-review/).
[`checks.json`](pr519-review/checks.json) records exact argv, UTC starts, exits,
log SHA256s and checked source commit. [`source-manifest.json`](pr519-review/source-manifest.json)
binds the four changed Rust source/test files. All commands below actually ran
on the checked source, before push:

| Command | Result |
|---|---|
| `cargo fmt --all -- --check` | PASS, exit 0 |
| `cargo check -p memra-tier -p memra-kv --offline --all-targets` | PASS, exit 0, macOS compile |
| same check with `--target x86_64-unknown-linux-gnu` | PASS, exit 0, **cross-compile only** |
| `cargo test -p memra-tier -p memra-kv --offline --no-fail-fast` | PASS, **230 tests total**, 0 failed/ignored |
| `cargo clippy -p memra-tier -p memra-kv --offline --all-targets --no-deps -- -D warnings` | PASS, exit 0 |
| `git diff --check` | PASS, exit 0 |
| `bash tools/check-flags.sh` | PASS; 864 runtime literal reads, no uncovered names |

Counts: memra-kv 59; memra-tier unit 2, bank 43, contracts 45, peer 18,
placement 6, storage 53, doctests 4. Seven new regression tests ran, including
actual child-process lifecycle tests. `before-storage.log.gz` preserves the three
initial failures against the original implementations; `before-upgrade.log.gz`
preserves the deterministic lock-window failure with the original conversion
sequence extracted into the private test-hook helper (other fixes do not affect
that lock schedule). `tests.log.gz` includes the passing final regressions and suites.

The existing, unrelated `memra-gguf/src/source.rs:20` unused `AsRawFd` import
warning remains in dependency build output. No source outside the authorized
storage code/tests was changed to suppress it; requested package Clippy passed.

No GPU/nvcc is available here. No Linux execution, O_DIRECT execution, GPU/native
serving check, power-loss durability test, or performance measurement ran. These
CPU fixes do not substitute for the lane's outstanding hardware qualification.
No frozen `contracts.rs` trait semantics, dependencies, runtime env reads,
numerical programs, schedules, or generated boards changed.

The initial worktree was clean. During verification, the unrelated untracked
`research/spill-lead-20260919/rented-5090-20260919/receipts-box2/` directory appeared;
it was preserved and excluded from all review-fix commits. Only the requested
review note and its raw receipts were added outside A's source/test paths.

Push uses `core.hooksPath=tools/hooks`, without any bypass. Exact pushed remote
SHA and hook result are reported in the handoff (the final receipt-only commit
cannot truthfully self-reference its own hash).
