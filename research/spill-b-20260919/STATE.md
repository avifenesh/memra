# WP-B day 13 checkpoint (provisional, cells in flight): prefix eviction must credit admission and the driver
- Branch lane/spill-b-20260919: 1cfdac6eb merges origin/main ea08bc7f8; b351d7db9 gate (tools/prefix-evict-reclaim-gate.py); f4350c241 fix (worker.rs settle_reclaimed_prefix_bytes); a1a2239e3 gate calibration fix; ebda75396 TESTING.md section.
- Push refused by tools/hooks/pre-push perf-ci freshness ("engine files touched after the last perf-ci battery", the merge's engine files). No override; the lead pushes.
- Defect: reclaim-on-defer drops prefix planes (stream-ordered cuMemFreeAsync into the pool) and re-reads headroom in the same tick; the receipt line took its "before" AFTER the eviction, so it could never show the credit (#346). The idle-box drain arm only runs with no active peer (#445 shape on a busy box).
- Fix: snapshot pools before eviction; after it, fence model-owned streams, cuMemPoolTrimTo(used + cached_before) per device, one `[admit-oom] reclaim settle` line in bytes; the reclaim line's "before" moves before the eviction. No numeric-program change, no new MEMRA_* read, VMM door untouched.
- Local: cargo test -p memra-server -p memra-kv --offline under CPUQuota=1200%: 734 + 64 passed, 0 failed. fmt, check-flags, boundary, diff --check clean.
- BOX3 (/root/wt-b at ebda75396; bins under /root/spill-receipts/b-day13/bins/{main,fix}): main ea08bc7f8 sha 6fc3ec03..., fix f4350c241 sha 24d6b453....
- gate-main (first cell): REFUSED by the gate's own precondition after a complete calibration boot (cost-line lookup); kept as refused. gate-main-rerun in flight; gate-fix next.
- Receipts mirror target: research/spill-b-20260919/pro-single-day13/ (from /root/spill-receipts/b-day13/).
