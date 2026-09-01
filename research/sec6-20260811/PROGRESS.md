# cx-sec6 progress

- Branch/worktree: `lane/cx-sec6` at `/home/avifenesh/projects/wt-cx-sec6`
- Starting revision: `126e6642a11727d7f0ee95254ef3dc2c17cf289f`
- Scope: bound abandoned constraint-compile workers per model, document compile isolation, and
  catalog the concluded `MEMRA_OPTI_CONTROLLER_Q` door.
- Required gate after each item: `nice -n 15 taskset -c 0-7 cargo test -p memra-server --release`.
- Constraints: CPU-only; no GPU runtime, benchmark, `cargo fmt`, merge, tag, push, or
  performance-board change.

## Status

- [x] Lane brief, repository law, and current Rust thread-lifecycle documentation read.
- [x] Fresh CPU-only crate baseline: 181 passed, 0 failed.
- [x] Item 1: per-model fail-closed cap for abandoned constraint-compile workers.
- [x] Item 2: constrained-decoding compile-isolation documentation.
- [x] Item 3: `MEMRA_OPTI_CONTROLLER_Q` FLAGS catalog row.
- [x] Final diff and four-commit audit.

## Evidence

- The branch started clean at the expected `126e6642` base.
- Item 1 pre-fix regression: `compiler_refuses_work_after_four_outstanding_runaways` failed
  because the fifth compile was accepted after four timed-out workers remained blocked.
- Item 1 focused post-fix gate: both compiler regressions passed; the scheduler-progress timeout
  regression also passed.
- Item 1 full crate gate: 182 passed, 0 failed, 0 ignored.
- Item 2 documents the off-tick bounded queue, schema envelope, retryable watchdog timeout, and
  four-worker fail-closed cap, with sec5/sec6 evidence links.
- Item 2 full crate gate: 182 passed, 0 failed, 0 ignored.
- Item 3 catalogs `MEMRA_OPTI_CONTROLLER_Q` as fresh-process-only, `[0,1]`, default-absent, and
  NO-GO, with the opti2 revival gates; its now-covered drift-baseline entry was removed.
- Item 3 `tools/check-flags.sh`: green, no new drift.
- Item 3 full crate gate: 182 passed, 0 failed, 0 ignored.
- Final branch shape: one opening progress commit plus one independent commit per lane item.
