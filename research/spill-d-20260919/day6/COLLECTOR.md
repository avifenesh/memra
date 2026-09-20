# Day 6 — collector correctness milestone

Starting revision: `54dd7d4a` (the prior session's main integration and two
unfinished collector/runbook files were retained). No numerical program changed.

- `--execute` preserves every child argument, including literal `--` and child
  `--execute` options. The A2 runbook uses the full existing libtest name.
- Exit 2 plus a last-line `REFUSED:` or `Error:` diagnostic records `refused` and
  that exact line. Other failures stay failures; refusals never qualify hardware.
- The 250 ms sampler asks for `power.limit` and `power.max_limit`. Every capture
  and completed CELL retains all observed device/limit pairs (including N/A);
  integrity validation checks them against the hash-bound raw CSV. Missing/empty
  telemetry yields an empty list, never an invented default.
- Fetched A's branch at `fd49a668ca30a63808c7bad6e7612c38445c3b82`:
  `research/spill-a-20260919/` has no `storage-cell` validation fragment.
  Its LEAD-FRAGMENTS.md covers the older telemetry join only. Carry forward the
  strict capture/hash/run-id storage-cell validator until A supplies its contract.

Ran `python3 -B -m unittest discover -s crates/memra-tier/tests/battery -p
 'test_*.py'`: **51 tests passed**; raw output `collector/python.log`.
`python3 -m py_compile tools/tier-battery.py` and `git diff --check` passed.
These are CPU/stub checks, not GPU/storage/serving qualification.
