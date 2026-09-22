# Session C day 24: the 5090 door gates on the slice-1 tree, the capture-isolating cell stopped at its arithmetic, the slice-3 census

Lane `lane/spill-c-20260919`, checkout `wt-spill-c`. Every push today in the announced
`MEMRA_RELEASE_QUALIFICATION_MODE=development` mode (the #589 hook refuses the tree `UNQUALIFIED` because the
content-bound census sees the engine files A's slice brought in; the hook prints `UNQUALIFIED DEVELOPMENT:
refs/heads/lane/spill-c-20260919 at <sha>; no GPU qualification claimed` and records the skip in the clone's
`.git/memra-gate-skips.log`). Every cell below is `executed-not-qualified` development evidence; no timing here is a
qualification claim. No commit on main, no PR. No engine change of my own today.

## Merge (first action)

`origin/main` `4bb2afb63` (#633, integ35: B day 29, the twin gate refuses typed on a broken V3 premise) merged as
`5951edf43`, clean; then `origin/lane/spill-a-20260919` `62aa92279` (A day 20, Move 2 slice 1 under the door, A's
own merge of main at day 21 included) merged as `fad129042`, clean, no conflict. After the merges `git diff
origin/lane/spill-a-20260919 -- crates/` is EMPTY and `git diff origin/main -- crates/ docs/FLAGS.md` is exactly
A's slice (`worker.rs`, `memra-tier` `conformance/d2d_capture.rs`, `contracts.rs`, the CPU bindings, the FLAGS door
row's day-20 sentence): this lane carries no engine line that A's tip does not. `research/INDEX.md` carries both
rows (`spill-a-20260919/day20`, `spill-c-20260919/day23`). `tools/check-conflict-markers.sh` OK. Pushed
(`fad129042` = `origin/lane/spill-c-20260919`).

## Task 1: the 5090 door gates on the slice-1 tree (owed by A since day 20)

Tree `11df6e653` (`crates/` equal to A's `62aa92279`), release `memra-server` `e0c8b09fb9982531...` built locally
under `systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G` (`day24-cpu/build-local.log`, `Finished
release in 2m 58s`, `rc=0`). Local RTX 5090 Laptop GPU, the Qwen3.5-9B NVFP4 MTP artifact for the host-tier gates
and the hit gate, the Qwen3.8-27B NVFP4-Q5K MTP artifact for the twin gate (the 9B twin refuses `cohort promotion
did not happen`, A's day-17 shape fact), `MEMRA_HOSTGATE_CACHE_MB=64`. Driver `day24-cell.sh` (the day-22 driver
plus `hit-{off,on}`, `twin27-{off,on}`, `unit-server`, `unit-engine`; every gate takes the canonical lock
`/tmp/memra-5090.lock` itself, the unit cells under the driver's own `flock -n`; a busy lock is retried 15 x 120 s
and the holder is never inspected or signalled; compute apps and driver free sampled before and after each cell),
`day24-local-battery.sh` (sixteen cells in one sitting). The card was idle at the start (no compute app, 23970 MiB
free). Receipts `rtx5090-day24/<cell>/` (`CELL.txt`, `gate.log`, `ev/`, `compute-apps.{before,after}.csv`,
`card.{before,after}.csv`, `verdict.txt`, `gate.exit`), `rtx5090-day24/battery.log`.

Sixteen cells 05:59:41Z to 06:11:39Z plus the rerun at 06:13:46Z (`battery.log`); one bounded lock retry
(`failure-default-off/lock-retries.txt`: `attempt 1: lock busy, waiting 120 s`; no compute app in its before
snapshot; the holder was never inspected or signalled); no compute app in any of the 34 before and after snapshots.
Every line verbatim; pass/fail cells, no timing claim.

| Cell | Environment | Verdict line | ok / FAIL |
|---|---|---|---|
| fault-default | default (door ON by construction) | `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` | 65 / 0 |
| fault-plain | `MEMRA_SERVE_SPEC=0` | `KV-HOST-CONTRACT-FAULT GATE: 2 FAILURE(S)` | 63 / 2 (the finding below) |
| fault-plain-rerun | `MEMRA_SERVE_SPEC=0`, the same cell again | `KV-HOST-CONTRACT-FAULT GATE: 2 FAILURE(S)` | 63 / 2 (same two lines, receipts `seq=3` and `seq=4` again) |
| failure-default-off | default | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` | 15 / 0 |
| failure-default-on | default, `MEMRA_KV_HOST_CONTRACTS=1` | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` | 15 / 0 |
| failure-plain-off | `MEMRA_SERVE_SPEC=0` | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` | 15 / 0 |
| failure-plain-on | `MEMRA_SERVE_SPEC=0`, door ON | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` | 15 / 0 |
| identity-default-off | default | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` | 12 / 0 |
| identity-default-on | default, door ON | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` | 12 / 0 |
| identity-plain-off | `MEMRA_SERVE_SPEC=0` | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` | 12 / 0 |
| identity-plain-on | `MEMRA_SERVE_SPEC=0`, door ON | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` | 12 / 0 |
| hit-off | door OFF (`spec-on-cache-hit-gate.sh qwen`, 9B) | `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` | 61 / 0 |
| hit-on | door ON | `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` | 61 / 0 |
| twin27-off | door OFF, the 27B, gate defaults (1024 MiB budget) | `PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=736755712 turns=8 cold_turns_after_1=0 cached_ok=7/7 lines_ok=8/8 evictions=9 cohort_evictions=3 self_evictions=0 refused_or_skipped=0 effective_free_ok=8/8 identity_ok=8/8 grid_ok=21/21 grid=32 off_grid_calls=0 V1=ok V2=ok V3=ok V4=ok V5=ok V6=ok -> PASS` | PASS (no `REFUSED: V3 premise` line: the premise held on a clean card) |
| twin27-on | door ON, the 27B | the identical line, `-> PASS` | PASS |
| unit-server (`option_b_*`, `option_c_*`, ignored GPU cells) | the door's unwinds on the card | `test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 805 filtered out; finished in 0.35s` | 8 / 0 |
| unit-engine (`d2d_capture_*`) | the capture class on the card | `test tier_transfer::tests::d2d_capture_lands_on_the_copy_stream_and_publishes_only_after_its_event ... ok`, `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 553 filtered out; finished in 0.30s` | 1 / 0 |

Where the route engaged on this card: every plain-arm seed in the identity, failure and fault logs prints `capture
submitted off the tick (seed): 64 tokens, 16 planes (53.6MB) on the contracts door's copy stream; recurrent state
cloned at the boundary on the owner stream` then `capture published off the tick (seed): 64 tokens complete after
1 poll(s), 0.0ms from submission to completion, 0.0ms to publication (tick-top poll)` (the 9B's 16 planes are about
0.9 MB of the 53.6 MB entry); the default (spec) arms publish through the spec boundary and show no capture line,
as on A's target card. The twin gate's `-> PASS` on both arms reproduces the lead's integ35 clean-card line to the
byte (`evictions=9 cohort_evictions=3 ... effective_free_ok=8/8`), now on the slice-1 tree. Fourteen of the sixteen
owed cells are green on this card; the fault gate's plain arm is red on a gate clause, twice, with the receipt.

### Finding: the fault gate's plain arm reads `2 FAILURE(S)` on the slice-1 tree, and the two lines are the gate's seq pin

`fault-plain` (`MEMRA_SERVE_SPEC=0`, the door ON by construction) printed, verbatim:

    FAIL: presubmit: the next demote completes with a D2H contract receipt after the refusal
    FAIL: postpublish: the next demote completes with a D2H contract receipt after the refusal
    KV-HOST-CONTRACT-FAULT GATE: 2 FAILURE(S)

63 `ok` against day 23's 65 on main's tree (`rtx5090-day23/fault-plain`, `ALL GREEN`); the other seven clauses of
both cells `ok`, the four promote cells `ok`, no compute app in the before or after snapshot. The server logs say
what moved. On main's tree the plain arm's seed publishes on the tick and the first demote's ticket is `seq=1`
(presubmit) or `seq=2` (postpublish). On the slice-1 tree the plain arm's seed takes slice 1's route: every seed
prints `[prefix-cache] capture submitted off the tick (seed): 64 tokens, 16 planes (53.6MB) on the contracts
door's copy stream ...` and its ticket consumes a sequence number on the same issuer, so the demote after the
injected refusal completes as `[prefix-host] contracts door D2H receipt: ticket issuer=2 seq=3 epochs=0/1/1 items=16
(8 KV planes) complete=16 require=ok ... retired acknowledged` (presubmit; `seq=4` in postpublish), then `demote
published off the tick: ticket seq=3 complete after 1 poll(s), 24.1ms ...` and `demote: 64 tokens, 54.6MB in
46.3ms`. The gate's `cell` clause (`tools/kv-host-contract-fault-gate.sh` line 203, `cell presubmit ... 1`, `cell
postpublish ... 2`) matches `contracts door D2H receipt: ticket issuer=[0-9]+ seq=$seq .* require=ok` with the
literal expected seq, so the receipt with `seq=3` does not match and the clause reads FAIL while the demote it
asks about completed with `require=ok`. The `pcell` promote clauses carry no seq pin and passed. The default (spec)
arm passed because under spec the seed publishes through `prefix_insert_from_spec_boundary`, the tick program,
which takes no ticket. Reading: a gate assumption (the first demote holds ticket seq 1) that slice 1 broke on the
plain arm, not an engine failure; A's day-20 target-card fault gate ran the default arm only (`contract-fault`, one
cell), so the plain arm under slice 1 is first seen here. Recorded for the lead and lane A; the fix (the clause
matching the first receipt AFTER the refusal line rather than a literal seq, or the seq read from the `demote
submitted off the tick: ... ticket seq=N` line) is A's or the lead's; no gate line changed on this lane today.
`fault-plain-rerun` below is the same cell a second time in the same sitting.

## Task 2, pre-registration (committed before any run; the arithmetic below stops the cell at this section)

**What A's cell could not resolve.** A day 20 (`stall_cell.py --mode capture`): the intruder's 5120-token prime is
on the tick in both arms (`stall_median` 355.4 against 355.4, 354.0 against 353.9, `flat`), so the capture's
share is under the cell's resolution. The owed cell isolates the capture: the intruder's prime near zero, its
publish a fresh entry.

**The shape as asked, and what the code offers.** A retire that "captures a fresh entry from a generated suffix"
does not exist as a publish class on this tree. `prefix_insert_from_session` is called at two sites only, `seed`
(`maybe_prefix_seed`, the prompt-end grid seed at prefill-done or at the armed grid boundary) and `lcp-split` (the
prime-time boundary of a covering entry); the spec path publishes at `spec-boundary`, `dspark-boundary` and
`glm5-boundary`, all captures of a prime stop; `maybe_plain_checkpoint` takes a `CacheSnapshot` for plain-affinity
rollback, not a prefix entry. Generated rows are never published. The nearest shape that meets the intent is the
seed-deepen on a warm hit: `prefix_seed_deepens` publishes when the session's primed depth exceeds the deepest
covering entry by `PREFIX_CACHE_MIN_TOKENS` = 64 tokens, and the seed publishes at the grid boundary (32). So: an
untimed setup post publishes E0 from a prompt P of 5120 tokens (on grid); the timed intruder is P plus exactly 64
fresh tokens (5184, on grid, `max_tokens=1`): a device hit on E0 (whole-entry restore), a prime of 64 rows, and the
seed of a fresh 5184-token entry E1 whose KV planes are the door's term. A's day-20 ON log shows this route firing
once by accident: `insert (seed): 5120 tokens, 308.9MB` on the one re-post whose prompt tokenized to exactly 5120.
Boot, harness and rules would be A's day-20 cell with that intruder (`MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4
MEMRA_SERVE_SPEC=0 MEMRA_PREFIX_CACHE_MB=1024 MEMRA_KV_HOST_MB=8192`, door OFF and ON, N=5 per arm per order, both
orders, one collector hold on the target card, 250 ms telemetry, C1's 6.0 ms threshold, `errors=0` and
`tenant_text_identical=True`, every intruder `cached_tokens=5120` and every re-post of P plus 64 hitting 5184).

**The claim, as it would have been registered.** With the door ON the tenant's stall median for a capture drops
toward the OFF-idle tick and the ON arm's `server_capture_ms` moves off the tick (the copy's duration observed at
the tick-top poll, not in the tenant's ITL).

**The arithmetic, on the receipts, that stops the cell.** Bytes per token of the 27B's trunk KV planes, from the
two entry sizes in the banked logs: (308.0 MB at 5088 tokens minus 158.8 MB at 64 tokens) / 5024 tokens = 29.7 KB
per token; the fixed part (recurrent state plus `last_h`) 156.9 MB. The term the door moves off the tick is the KV
plane copy of E1 only: 5184 x 29.7 KB = 154 MB; at the served context's largest entry (8192 tokens) 243 MB. Both
arms keep on the tick: the whole-entry restore of E0 (152 MB of KV planes plus the 157 MB state, D2D, the restore
program), the 64-row prime chunk (on this card the day-23 promote OFF arm read a stall median of 85.2 ms for a
64-token host hit with a 22-to-25-row prime, a 10.5 ms H2D and a 6 ms inline demote, so one prime chunk of a few
dozen rows is tens of milliseconds), the `clone_dtod` of the 157 MB state on the owner stream, 32 `alloc_u8`
(`cuMemAllocAsync` plus a memset each), and from the fourth intruder of each boot (1024 MB budget, 311 MB entries)
the eviction demote of E1's victim (OFF: the synchronous D2H, A's OFF `server_demote_ms` 68 to 76 ms at 309 MB; ON:
Move 1's submission, 228 ms to the observing poll in A's cell). The moved term is a device-to-device copy of 154
MB: no D2D bandwidth receipt exists in this repository for this card, so the bound is stated against a figure an
order of magnitude below the card's memory bandwidth class (about 1.8 TB/s, a copy paying read and write): at 100
GB/s the copy is 1.5 ms (2.4 ms at 8192 tokens), under C1's 6.0 ms threshold, and at anything above 110 GB/s it is
under the harness's own resolution (idle p99 minus p50, 14.8 minus 13.4 = 1.4 ms in every day-16 to day-23
receipt). So the shape cannot make the capture the dominant term: the door-moved bytes are bounded by the served
context to a copy the cell cannot see, while the terms both arms share are tens of milliseconds by receipt. The
pre-registered claim would read `flat` by construction and decide nothing the arithmetic does not already say. The
cell stops here: no box time taken, no run, no verdict. What WOULD make the capture visible is not a shape but a
different term: the eviction demote's on-tick share under ON (A's 228 ms against OFF's 70 ms) is Move 1's owed item
(the two 160 MB hashes at submission), not slice 1's, and is already named in the review table's arithmetic.

## Task 3: the slice-3 and spec-boundary census

`HOSTPREFIX-DOOR.md`, review table section D, item 10: which publishes still take the tick program after slice 1
(`prefix_insert_from_spec_boundary` from the MTP drain sweep, the dspark and glm5 publishers; the fanout leader and
pause sweep snapshots; slice 1's own `OnTick` refusals), the bytes each moves on the 27B at the served context
(trunk 29.7 KB per token, 243 MB at 8192; the MTP draft plane about 1.9 KB per token, 15 MB at 8192, from the 158.9
against 158.8 MB pair at 64 tokens; the state and boundary logits owned by the `SpecBoundaryCapture`, no copy at
publish; the DSPARK tail about 85 MB fixed), and what the off-tick route would need (a second borrowed span, the
draft scratch owned by `memra_engine::spec::SpecSession`; a producer event at the drain sweep; the borrow's
lifetime across a retiring or parking spec session, slice 1's settle-before-drop extended; slice 3's receipt term
over both plane classes). No code.

## Hygiene on the final tree

`cargo fmt --all -- --check` rc=0 (`day24-cpu/fmt-merge.log`); `cargo test -p memra-server --lib` `799 passed; 0
failed; 14 ignored` (`day24-cpu/server-tests-merge.log`); `cargo test -p memra-tier` every binary `0 failed`, the
contracts `74 passed` (`day24-cpu/tier-tests-merge.log`); the two test binaries prebuilt under the CPU quota before
the unit cells (`day24-cpu/test-build-{server,engine}.log`) so the lock hold carried the run, not the build;
`tools/check-flags.sh` no uncovered runtime names (no new `MEMRA_*` read: the drivers set existing names);
`tools/check-conflict-markers.sh` OK; `python3 tools/check-public-boundary.py check` 0 new (the receipts carry
`127.0.0.1` only); `git diff --check` clean; no em dash in any file written today; shellcheck on the two drivers
info-level only (SC2094 on the lock fd's own path).

## Pushes

`fad129042` (the two merges), `11df6e653` (the drivers before the run), `dd91c4949` (twelve cells' receipts, this
file with the finding and the pre-registration, the census), then the closing commit (the remaining receipts,
STATE, INDEX), each in `MEMRA_RELEASE_QUALIFICATION_MODE=development` (printed `UNQUALIFIED DEVELOPMENT ... no GPU
qualification claimed`, logged in the clone's `.git/memra-gate-skips.log`). Not merged into main, no PR opened.

## Left as it was, and cleanup

Not touched: BOX3 (no connection opened today: Task 2 stopped at its arithmetic and Task 1 is the local card's),
`/root/artifacts`, `/root/memra-spill`, other lanes' worktrees or processes (the local lock was busy once behind a
holder with no compute app, seen only in the retry record; another session's leftover shell matched a `pgrep` of
mine and was left alone). No `/tmp` scratch created (the battery wrote under `rtx5090-day24/` and `day24-cpu/`
directly); no server of mine on the card at close (compute apps empty, 23970 MiB free, 56 C); both locks free.

## Budget

About 1.3 agent-hours against 4: reading and the two merges 0.2, the drivers and the build 0.2, the sixteen cells
plus the rerun 0.3 (wall 05:59Z to 06:15Z, the cells ran while the census was read), the fault-plain diagnosis
0.2, the capture cell's code reading and arithmetic 0.2, the census and records 0.2. Blockers: none. Open for the
lead and lane A: the fault gate's `cell` clause pins the ticket seq (1, 2) that slice 1's plain-arm captures now
consume; the capture-isolating cell is stopped by arithmetic, not by the box; the spec-boundary route census is in
the door review's item 10.
