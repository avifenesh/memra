# Self-review: integ23 (C day 17: #586 fix and the arena pair cell; B day 21: #523 map, #427 classification, #372 verified landed)

Author's review of the full diff `main..lane/spill-integ23-20260921`, posted as a PR comment per the owner rule.

## What the diff is
- `tools/memra_cpu_experts.cpp`: one hunk. The per-job `fetch_sub` on `prefetch_inflight` is removed and a single
  `fetch_sub` sits inside the final-half branch, so a mirrored projection (two I/O jobs) releases its one charge
  once, on success or failure. The clamp in the public stats function stays and its comment now says it is not the
  balancing oracle. I read the surrounding `worker_loop` completion path: the final-half branch is the only place
  the projection is known complete, and the failure path enters the same branch.
- `tools/memra_cpu_expert_prefetch_test.cpp` and `tools/test_cpu_expert_prefetch.sh`: the fixture includes the
  production translation unit (the shm test's existing pattern), renames its one `pread` so a cell can hold a half,
  and reads the signed counter directly. Cells `barrier`, `failure`, `parity`. It refuses to run where `/dev/shm` or
  `O_DIRECT` is unavailable rather than skipping (stated in `docs/TESTING.md`), so CI cannot pass it by skipping.
- `.github/workflows/ci.yml`: one step in the engine-tests job after `cpu_native_check`. The workflow-key census
  passes (no duplicate mapping keys), which is the failure mode #600 fixed.
- `docs/TESTING.md`: the fixture's paragraph.
- Research: C's DAY17, arena cell scripts and receipts, cpubank receipt, WC pointer edits; B's DAY21, bisect and
  gate runners, receipts from both cards; INDEX rows; the lead record section and this file.

## What I checked
- No `crates/` change; `git diff --name-only main HEAD | grep '^crates/'` is empty. No new `MEMRA_*` read (flags
  census clean). Lock names in the new scripts: `/tmp/memra-gpu.lock` and `/tmp/memra-5090.lock` only.
- The #586 fixture ran red on the base and green on the fix in C's receipts (`day17-local/`), and it runs again in
  this integ's CPU battery on the merged tree (`integ23-cpu-battery/cpu-prefetch-586.log`).
- The arena replay (`arena-pair.py`) re-derives the verdict from the mirrored logs in this battery.
- B's day changed no code; its gate runs are receipts on both cards, replayed by their own scripts on the lane.
- Public boundary `check`: 0 new. No em dashes in added prose (scan in the battery). INDEX.md carries no marker
  line of any of the four kinds.
- Issue actions on merge (lead): close #586 (fix plus fixture in CI), #523 (four items held, map on the issue),
  #372 (landed by #377, verified today); #427 stays open (classified, kernel unnamed, fix owed); #385 with A.

## What I did not do
- No GPU cell of my own; no serve smoke on this integ (no engine change; the CI step is CPU-only).
- Not verified here: the new CI step on a GitHub runner (this PR's CI run is that check).
