# cx-sec5 progress

- Branch/worktree: `lane/cx-sec5` at `/home/avifenesh/projects/wt-cx-sec5`
- Starting revision: `0e890ccf72c539e040624dd3c3784586c7e419ed`
- Scope: three independent CPU-only serving-hardening follow-ups from the sec3/sec4 review.
- Required gate after each item: `nice -n 15 taskset -c 0-7 cargo test -p memra-server --release`.
- Constraints: no GPU runtime, benchmark, `cargo fmt`, merge, tag, push, or performance-board move.

## Status

- [x] Lane brief and repository law read.
- [x] Fresh baseline: 178 tests passed, 0 failed.
- [x] Item 1: compile-time watchdog and queue-drain regression.
- [x] Item 2: per-tenant ADSD historical self-baseline fallback.
- [x] Item 3: operator-only global LCP/prefix aggregate counters and serving docs.
- [x] Final diff and three-commit audit.

## Evidence

- Item 1 pre-fix regression: `compiler_abandons_runaway_job_and_drains_next_job` failed because
  job 2 remained behind a deliberately stuck job 1.
- Item 1 focused post-fix regressions: watchdog queue drain 1 passed; normal-decode/heartbeat
  progress during timeout 1 passed.
- Item 1 full crate gate: 179 passed, 0 failed, 0 ignored.
- Item 2 pre-fix regression: the single-tenant synthetic collapse emitted 0 incidents instead of 1.
- Item 2 focused post-fix ADSD gate: 4 passed, 0 failed.
- Item 2 full crate gate: 180 passed, 0 failed, 0 ignored.
- Item 3 pre-fix regression: a completion credential still received `lcp_histogram`.
- Item 3 focused post-fix metrics/auth gate: 7 passed, 0 failed.
- Item 3 full crate gate: 181 passed, 0 failed, 0 ignored.
- `git diff --check 0e890ccf72c539e040624dd3c3784586c7e419ed..HEAD`: clean.
- Branch audit: exactly three conventional `fix(server):` commits, one per lane item.
