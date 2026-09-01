# cx-cachemeter — regression isolation progress

Branch: `lane/cx-cachemeter`

Original pushed base: `56f7ac0d8a20c367d6dba25cc03427098bf7f248`

Primary target after steering 2: `43caa7e1213167e685012b368479ead4e1dc9850`
(`keep/cx-eosclass-merged`, SLRU plus the B1FAST repair; not pushed)

Independent-verification target after steering 4:
`96361c531d26bdd95f4330617b872b2fa7d96f3d` (the local prefix-snapshot fix on top of
`43caa7e12`; not pushed).

Preserved comparison tips:

- `keep/cx-budgetsize-merged`: geometry-derived prefix-cache budget and admission metrics/warning.
- `keep/cx-shmconflict-merged`: the budget-size merge plus the `cu/flash_attn.cu` bank-conflict change.

## Scope and stop condition

Isolate the first commit and hunk that changes the cache-metering and Q35 mixed-c=4
`serve-smoke` arms from pass to fail, fix the smallest responsible behavior without weakening the
gate, run the mandatory local validation battery, commit the evidence and fix on this lane, and
stop. Do not merge, tag, push, edit generated performance boards, bypass hooks, or run
`cargo fmt`.

Plain `56f7ac0d8` did fail both requested arms, so forward attribution to the budget-size and
shared-memory-conflict merges stopped. Steering 2 subsequently resumed the lane for one bounded
question: rebase the evidence commits onto `43caa7e12`, then test that now-primary SLRU+B1FAST tree
with the shipped policy and `MEMRA_PREFIX_CACHE_POLICY=lru`. Do not test either held merge unless
both policy arms pass there.

Steering 4 supersedes that stop condition for independent verification. Rebase the evidence-only
lane onto `96361c531`, run the cache-metering and Q35 hot-seed cells under both policies, then test
the budget-size and shared-memory-conflict descendants on top of the fix. Do not change the fix or
the gates; any failure is a verification result.

## Evidence rules

- Capture raw command output with `tee` before parsing it.
- Quote failure causes; do not infer uncaptured causes.
- Preserve `/tmp/battery-perfci-20260813.log` as the initiating primary evidence.
- Serialize the full quick-performance battery under `flock /tmp/memra-5090.lock`.
- Never weaken thresholds, skip required arms, or increase the configured cache budget to hide an
  admission refusal.

## Test matrix

- [x] Plain `56f7ac0d8`, shipped/default SLRU: **FAIL** in both focused arms.
- [x] Plain `56f7ac0d8`, explicit `MEMRA_PREFIX_CACHE_POLICY=lru`: **FAIL** with the same
  zero-insert signature.
- [x] `43caa7e12`, shipped/default SLRU: **FAIL** in both focused arms.
- [x] `43caa7e12`, explicit `MEMRA_PREFIX_CACHE_POLICY=lru`: **FAIL** with the same
  zero-insert signature.
- [ ] `keep/cx-budgetsize-merged`: intentionally not run; the baseline-first stop condition fired.
- [ ] `keep/cx-shmconflict-merged`: intentionally not run; the baseline-first stop condition fired.
- [x] Exact refusal hunk isolated in source/history: `c3976d488`, which added the
  `l.len != cache.pos` hard error over every allocated KV slot. A pre/post GPU first-bad run was
  intentionally not started after the stop condition.
- [ ] Minimal fix: intentionally not attempted after the baseline-first stop condition.
- [ ] `cargo test -p memra-server`: intentionally not run; no merged-tree fix exists to qualify.
- [ ] Full `tools/local-ci.sh --perf-quick`: intentionally not run; both mandatory focused arms are
  already known red on the pushed base.
- [x] `RESULTS.md` and raw logs prepared; lane stopped without merge/tag/push/board edit.

### Steering 4 independent-verification matrix

- [x] `96361c531` plus evidence only, shipped/default SLRU: both focused arms PASS.
- [x] `96361c531` plus evidence only, explicit `MEMRA_PREFIX_CACHE_POLICY=lru`: both focused arms
  PASS on the same binary.
- [x] `96361c531` plus `keep/cx-budgetsize-merged` changes: both focused arms PASS.
- [x] `96361c531` plus `keep/cx-shmconflict-merged` changes: both focused arms PASS.
- [x] Record actual cache-metering and Q35 counters for every cell.
- [x] Independent `cargo test -p memra-server`: 248 passed, 0 failed.

## Timeline

- 2026-08-13: Confirmed the worktree is clean on `lane/cx-cachemeter` at exact base
  `56f7ac0d8a20c367d6dba25cc03427098bf7f248`; both preserved merge branches resolve locally.
- 2026-08-13: Created this file as the first artifact in `research/cachemeter-20260813/`.
  Baseline execution has not started.
- 2026-08-13: Copied the initiating `/tmp/battery-perfci-20260813.log` byte-for-byte to
  `raw/battery-perfci-pre-reset-20260813.log`; both files have SHA-256
  `52b8a528a8b329a07013baade93393df343febf7c3828c812c3ef62ddc25f2de`.
- 2026-08-13T07:16:28+03:00 to 07:20:15+03:00: Under
  `flock /tmp/memra-5090.lock`, ran the two focused arms against the base engine with
  `MEMRA_PREFIX_CACHE_POLICY` unset. Cache metering failed with 0 inserts, 0 hits, and 6 misses;
  Q35 failed with `q35: cache retains 0 entries after 8 hot seeds`. The Q35 server did recognize
  the explicit 4096 MiB setting as `budget 4295MB` and reported the cache `on`.
- 2026-08-13T07:20:21+03:00 to 07:21:35+03:00: Repeated the same two arms with only
  `MEMRA_PREFIX_CACHE_POLICY=lru` changed. Both failed with the same zero-insert behavior. The
  release binary SHA-256 was identical in both runs:
  `d908cec3d5c12209150c40f8479860a763e056c1677cf9fe605b4205e6f7895f`.
- 2026-08-13: Captured the server's direct refusal in both policy arms:
  `prefix snapshot layer 32 len 0 != cache pos 272` for the 9B gate and
  `prefix snapshot layer 40 len 0 != cache pos 4860` for Q35, each followed by
  `prefix not cached`.
- 2026-08-13: Confirmed `git diff --quiet 56f7ac0d8..HEAD -- .
  ':(exclude)research/cachemeter-20260813'` exits 0. The lane commits add evidence only; all engine
  and existing harness inputs are exactly the requested base. Also confirmed `56f7ac0d8` is an
  ancestor of `keep/cx-budgetsize-merged`, which is an ancestor of
  `keep/cx-shmconflict-merged`.
- 2026-08-13: **Baseline-first stop condition fired.** The pushed base is already broken, so the
  two preserved merges did not introduce this zero-insert regression. Forward testing, a fix, and
  the full post-fix battery were not attempted in this lane.
- 2026-08-13: Steering 2 made local merge `43caa7e12` the primary test target. Rebased the three
  evidence-only lane commits from `56f7ac0d8` onto `43caa7e12`; the original base receipts remain
  immutable under `raw/base-*`. The next action is the exact same two-policy focused matrix on the
  SLRU+B1FAST tree.
- 2026-08-13T07:28:14+03:00 to 07:29:48+03:00: Under
  `flock /tmp/memra-5090.lock`, ran both focused arms on the `43caa7e12` engine with the policy
  unset. Cache metering failed 13 assertions with 0 inserts, 0 hits, and 6 misses. Q35 failed with
  `q35: cache retains 0 entries after 8 hot seeds`; its cell recorded 0 inserts, 0 hits, and 20
  misses. The server repeated the same layer-32/layer-40 snapshot refusals as the pushed base.
- 2026-08-13T07:30:02+03:00 to 07:33:38+03:00: Repeated the current-tree matrix with only
  `MEMRA_PREFIX_CACHE_POLICY=lru` changed. Both focused arms failed identically, and boot explicitly
  reported `policy plain-LRU`. This excludes the SLRU probation/promotion policy from the refusal:
  snapshot construction returns `Err` before either policy's insert path is called.
- 2026-08-13: Source/history isolation found the exact refusal check in `c3976d488` (`feat: restore
  transformer prefixes at LCP splits`). It requires every allocated `Some(KvLayer)` to have
  `l.len == cache.pos`. Both failing artifacts carry one NextN layer beyond the executed trunk
  (9B: 33/1, Q35: 41/1), so the first non-trunk KV slot is allocated but remains length zero; the
  captured failing indices are exactly 32 and 40. The `Err` arm logs `prefix not cached` and never
  calls `PrefixCache::insert`, where `self.inserts += 1` lives. No eviction policy can affect that
  pre-admission failure.
- 2026-08-13: Steering 4 supplied local fix `96361c531` and changed this lane from diagnosis to
  independent verification. Rebased all five evidence-only commits from `43caa7e12` onto the fix;
  `git diff --quiet 96361c531..HEAD -- . ':(exclude)research/cachemeter-20260813'` exits 0.
- 2026-08-13T08:11:27+03:00 to 08:14:36+03:00: Under the shipped default byte-SLRU policy, both
  focused arms passed on the fixed engine. Cache metering observed prompt/cached/computed
  1632/1024/608, hit ratio 0.6274509803921569, hits/misses/inserts 4/2/2, hit tokens 1024, and the
  asserted revenue multiplier 2.6842. Q35 observed 18 hits, 2 misses, 2 inserts, 87,480 cached
  tokens, no accounting drift, and 8 to 10 retained entries; all 20 responses were exact.
- 2026-08-13T08:14:48+03:00 to 08:15:25+03:00: Repeated both arms with
  `MEMRA_PREFIX_CACHE_POLICY=lru`; the server explicitly reported `policy plain-LRU` and every
  counter above repeated exactly. Both policy runs used byte-identical server SHA-256
  `59889be44b48c3c9b3ed19a229d265ab8fbcdf98777400cc266a7a1e5626b628`.
- 2026-08-13T08:15:35+03:00 to 08:18:37+03:00: Tested clean virtual merge tree
  `2d55b47d278319363e8fa3d957185841c2348b66` (fix plus `keep/cx-budgetsize-merged`) under the
  shipped SLRU policy. Both focused arms passed with the same cache-metering and Q35 counters as
  the fixed base. The 9B server exercised the new derived 348,651,520-byte budget; Q35 honored the
  explicit 4,294,967,296-byte budget. Server SHA-256:
  `c7dd155100bab78cdf7925abc7e5adb2a36ad370563824f45b718a65af70d84f`.
- 2026-08-13T08:18:54+03:00 to 08:19:37+03:00: Tested clean virtual merge tree
  `9310d8008700aca49de0add89a877ba799cc71ad` (fix plus
  `keep/cx-shmconflict-merged`, including its `flash_attn.cu` change) under shipped SLRU. Both arms
  passed with the same exact counter values and no snapshot-refusal/failure line. Server SHA-256:
  `d18408cde91374094eebe374366df36d334feaee2687081b2fafe83da0e2ef68`.
- 2026-08-13: Independently ran `TMPDIR=/home/avifenesh/tmp-lanes cargo test -p memra-server` on
  the fixed lane: 248 passed, 0 failed.

Historical diagnosis verdict: **ALREADY BROKEN AT PUSHED BASE; PRIORITY CHANGE REQUIRED.** The
independent verification verdict is now **PASS** for the fix under SLRU and LRU and for both held
descendants under shipped SLRU. The separately observed full-battery log has green correctness and
`serve-smoke: 0 failed`, but its perf stage records four cross-day tripwire failures; do not call
that full battery wholly green.
