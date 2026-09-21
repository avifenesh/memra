# Self-review: integ24 (A day 15: #385 arena startup on the target card, the door's cached-destination pair)

Author's review of the full diff `main..lane/spill-integ24-20260921`, posted as a PR comment per the owner rule.

## What the diff is
- `tools/pinned-host-reserve-bench.py`: the measurement harness gains the `chunked` (N threads, bytes/N chunks) and
  `thp` (madvise huge pages, register) arms, a `cuMemHostGetFlags` read-back per arm, the huge-page fraction from
  smaps, and the roundtrip correctness pass. It is a bench tool under `tools/`, not engine code; no `MEMRA_*` read.
- `research/spill-a-20260919/`: DAY15 (pre-registration committed before the runs, results, side effect, CPU gates),
  STATE, `arena-ab.py` replay, `HOST-ARENA-STARTUP.md` update, receipts for both arena cells and the WC pair cell
  (telemetry, receipt.json, driver and replay logs), INDEX row.
- `research/spill-lead-20260919/`: the day-12 record's integ24 section, this file, the battery summary and logs.

## What I checked
- No `crates/` change; no new flag; lock names in the new scripts are `/tmp/memra-gpu.lock` only.
- The rule did not move after the runs: the pre-registration section (committed `ad11f2a70`) carries the floor
  (1.10), the huge-page minimum fraction (0.9) and the void clause; the replay applies them and agrees with the
  harness on both cells. The `thp` arm's 3.1x is reported as void, not as a win, exactly as the clause says.
- The reserve size is derived from the box (MemAvailable minus the engine's headroom margin) and stated; the 288 GiB
  figure is not used on 88 GiB of RAM.
- No engine change landed; the landing conditions are named (hugetlb pool or page-cache regime, reserve plus Drop plus
  fall-back, the 2x B200 decision cell).
- The WC pair is C's harness run by A with three path changes, stated as such; banked as review input, no verdict.
- Public boundary `check`: 0 new. No em dashes in added prose. INDEX.md carries no marker line of any kind.

## What I did not do
- No GPU cell of my own; no serve smoke (no engine change).
