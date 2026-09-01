# cx-shmown progress

Date: 2026-08-13
Branch: `lane/cx-shmown`
Base: `6aba8b2e59bf417e1af689cb8ad11ad2f87d7f24` (`main`)

## Scope

Harden the CPU shared-memory expert cache so it never silently adopts an unsafe
POSIX shared-memory object, never turns an invalid persisted offset into a pointer,
and detects sampled payload tampering across restarts.

This is a CPU-only lane. Do not use box1, the local RTX 5090, or any GPU lock.
Do not run or claim `kernel-check`, `run-gen`, or `run-spec`.

## Acceptance gates

- [x] Prefer self-creation with `O_CREAT | O_EXCL`; accept an existing object only
      when `fstat` proves the current uid owns it and no group/other permission bit
      is set. Log an explicit refusal and use the private cache otherwise.
- [x] Validate every persisted row with overflow-safe
      `shm_offset + pool_bytes <= segment_bytes` before pointer construction.
- [x] Store and verify a documented sampled checksum for each persisted entry.
- [x] Test foreign-owner/permissive-mode refusal plus private fallback.
- [x] Test past-end and overflowing persisted rows as rejected misses with no OOB.
- [x] Test that the safe self-created path still produces a warm hit.
- [x] Build and run the relevant CPU-side tests.
- [x] Update `docs/FLAGS.md` and write `RESULTS.md` with the GPU battery explicitly
      marked not run.
- [x] Commit only the intended lane changes; do not merge, tag, push, or edit the
      generated performance board.

## Log

- 2026-08-13: Confirmed a clean worktree on `lane/cx-shmown`; branch and `main`
  both point to `6aba8b2e5`. Created this ledger before any implementation change.
- 2026-08-13: Implemented exclusive self-creation, exact effective-uid/`0600`
  adoption, checked persisted ranges (including overflow and the arena data floor),
  and v2 per-entry checksums. Entries up to 32 KiB are hashed in full; larger
  entries hash eight evenly spaced 4 KiB windows, including both ends (32 KiB
  maximum per entry, about 0.2 GiB for the historical ~6,700-entry warm set).
- 2026-08-13: Added `tools/test_cpu_expert_shm.sh` plus its in-translation-unit
  harness. The first local run passed actual same-owner `0644` and subordinate-uid
  `0666` refusal/private-fallback cases, restrictive-umask self-create and warm hit,
  past-end and overflowing row rejection/source re-read, and sampled-payload
  checksum rejection/source re-read. Raw case logs are under `raw/`.
- 2026-08-13: Production companion build passed; existing `cpu_native_check`
  reported `ALL GREEN`; four Rust `cpu_experts` tests passed; the entire focused
  shm suite also passed under ASan+UBSan. GPU visibility was disabled for Rust
  commands. `kernel-check`, `run-gen`, and `run-spec` were not run by design; see
  `RESULTS.md` and the raw logs.
- 2026-08-13: A final two-process `cpu_native_check` ran through the production
  companion ABI with shm enabled and a stable source fixture. The cold process
  persisted 15 entries; the next process reopened all 15 warm, recorded 24 cache
  hits, and performed zero demand projection reads while remaining `ALL GREEN`.
- 2026-08-13: Final scope audit found only the shm hardening, focused harness,
  flag documentation, and this lane's evidence files. Generated performance
  surfaces remain untouched. The lane is ready for its results commit and owner
  integration; no merge, tag, or push is authorized.
