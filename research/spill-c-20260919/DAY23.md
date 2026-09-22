# Session C day 23: the promote stall cell on main's tree (the parked-only wait), the demote arm beside it, and the HOSTPREFIX door review table in its final form for 2026-10-05

Lane `lane/spill-c-20260919`, checkout `wt-spill-c`. Every push today in the announced
`MEMRA_RELEASE_QUALIFICATION_MODE=development` mode (the #589 hook refuses the tree `UNQUALIFIED` because the
content-bound census sees the engine files main brought in; the hook prints `UNQUALIFIED DEVELOPMENT:
refs/heads/lane/spill-c-20260919 at <sha>; no GPU qualification claimed` and records the skip in the clone's
`.git/memra-gate-skips.log`). Every cell below is `executed-not-qualified` development evidence; no timing here is a
qualification claim. No commit on main, no PR. No engine change of my own today.

## Merge (first action)

`origin/main` `0713c1a79` (#627, integ32: A's day 18, the promote half of Move 1, plus the lead's three revuto
fixes: the probe's cold memo names the refused entry; a tenant purge clears the tenant's memo and releases the
worker's one-tick pin; the run loop waits boundedly on the command channel when the queue holds only requests
parked on the in-flight promote instead of spinning the owner thread through park-and-requeue ticks) merged as
`d2b8efbf2`. One conflict, `crates/memra-server/src/worker.rs`, at the two sites where my day-22 resolution of
A's day 18 against #622 met the lead's resolution of the same pair on main: main's side taken whole (rerere had
offered the day-22 resolution; discarded). After the merge `git diff origin/main -- crates/` is EMPTY: this lane
carries no engine line that main does not. `research/INDEX.md` merged clean with both rows. `tools/check-conflict-
markers.sh` OK before the commit. Pushed in development mode (`d2b8efbf2` = `origin/lane/spill-c-20260919`).

## Task 1, pre-registration (this section is committed before the run)

**Owed.** integ32 (`research/spill-lead-20260919/INTEGRATION-DAY12.md`, "Owed to the door review"): A's day-18
promote stall receipt (`stall_median=81.9`) was taken with the owner-thread spin that revuto round 2 on #627
removed; the wait changes timing, not bytes, so the promote stall cell is owed on the tree that carries the wait.

**Tree and rig.** Main's tree as merged, `d2b8efbf2` plus the day-23 drivers (the tree SHA is in the cell's
`CELL.txt`); `crates/` equal to `origin/main` `0713c1a79`. One RTX PRO 6000 Blackwell at its 600 W limit through
the canonical collector (`tools/tier-battery.py --rig pro-single --timeout 3600 --external-lock`, lock
`/tmp/memra-gpu.lock`, 250 ms telemetry, `CELL.jsonl` `executed-not-qualified`), the Qwen3.8-27B NVFP4-Q5K MTP
artifact, `MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 MEMRA_SERVE_SPEC=0 MEMRA_PREFIX_CACHE_MB=256 MEMRA_KV_HOST_MB=8192`
in both boots, `MEMRA_KV_HOST_CONTRACTS=1` in the ON boot only. Lane A day 19 may hold the box; the runner
retries the lock 75 x 120 s and never inspects or signals the holder.

**Driver and harness.** `day23-stall-cell.sh` is lane A's `pro-single-day18/stall-cell.sh` (itself the day-16
script with the receipts root moved) with three path changes and one shape change, stated: the worktree, the
receipts root and the server binary are arguments instead of `/root/wt-a` and `/root/spill-receipts/a-day18`;
the tree SHA, the harness digest and the compute apps before and after are recorded; and the two boots (door OFF,
then door ON) run inside ONE collector lock hold, where A held the lock once per boot, so OFF against ON is a
same-window pair on one tree today. Each boot runs the demote arm then the promote arm, as A's did. The harness
is `research/spill-a-20260919/stall_cell.py` as it sits on this tree after the merge, byte-identical to lane A's
tip `c96d5186`: the tenant streams a 20-token prompt for 160 tokens, the intruder fires at the tenant's 24th
token, order 1 (idle, arm) x 5 then order 2 (arm, idle) x 5, N=5 per arm per order, N=10 pooled, the rule line
fixed in the harness and recomputed by `--replay`. `day23-box-run.sh` waits for the build receipt and runs the
cell once through the collector. `day23-stall-reading.py` holds the thresholds below as constants and prints one
`DAY23 STALL READING` line per question after replaying every rule line through the harness; it is committed
before the run and not edited after it.

**Claim (integ32's, read as a rule).** The parked-only wait removed the owner-thread spin, so on this tree:
(a) the tenant's stall median for a promote under the door stays at or below day 18's 81.9 ms; (b) the promote's
own `server_promote_ms` (submission to publication) does not grow beyond the day-18 figure by more than the
tick's 2 ms bounded wait. References are same-box, other-sitting figures: A day 16 OFF promote `85.0`, OFF demote
`117.5`, ON promote `162.8`, ON demote `193.5`; A day 18 run 2 ON promote `81.9` with `server_promote_ms` first pair
`60.8, 61.9` and steady `25.9` to `26.4` (median `26.1`), ON demote `149.7` (day 17: `149.6`).

Rules, fixed before the run (stall medians in ms, the harness's `stall_median`):
- P1, promote ON against day 18: `at_or_under_day18` if `stall_median <= 81.9`; `within_wait` if `81.9 <
  stall_median <= 83.9`; `grew` above.
- P2, the promote's own window ON: `within_wait` if the median of `server_promote_ms` over runs 3 to 10 is
  `<= 28.1` (26.1 plus 2.0), else `grew`; the first pair's max against `63.9` recorded the same way
  (`first_pair_within_wait` / `first_pair_grew`).
- P3, promote ON against OFF in the same window: `on_at_off` if `on <= 1.10 x off`, else `on_above_off`.
- P4, promote OFF against day 16's 85.0: `off_stable` inside `76.5..93.5`, else `off_moved` (a regime reading of
  the OFF program, which the door does not touch).
- P5, demote ON against day 18's 149.7: `demote_half_unchanged` inside `134.7..164.7`, else `demote_half_moved`
  (the wait fires only with a `Promoting` entry, so the demote arm should not move).
- P6, demote OFF against day 16's 117.5: `off_stable` inside `105.8..129.3`, else `off_moved`.
- Admissibility (A's day-18 list, applied to the ON promote arm): `STALL REPLAY: PASS` for all four receipts,
  `errors=0`, `tenant_text_identical=True` in every receipt; in the ON boot's server log ten `promote submitted
  off the tick`, ten `promote published off the tick`, ten `contracts door H2D receipt` lines over the promote
  arm's ten intruders, every intruder `cached_tokens=64`, no `demote failed`, `promote failed`, `promote refused`
  or `TIER DISABLED` line, and zero `settled synchronously by a promote` lines. An inadmissible run is reported as
  such and not read against P1 to P6.
- Recorded, no rule: the `poll(s)` count and the submission-to-completion ms in each `promote published off the
  tick` line (day 18 read `1 poll(s), 19.5` to `19.7ms` in all ten).

What this cell is and is not: one card class, one host, one artifact, `MEMRA_SERVE_SPEC=0`, 64-token entries
(160 MB), N=5 per arm per order in both orders, one sitting; P3 is a same-window pair, P1, P2, P4, P5 and P6 are
same-box cross-sitting readings against A's receipts, not same-window A/Bs; no number is divided into a number
from another box. Nothing here answers the door's decision question.

## Task 1, the run (target card, one RTX PRO 6000 Blackwell Server Edition at its 600 W limit, collector rig `pro-single`)

Tree `91b0d4e08` (`crates/` equal to `origin/main` `0713c1a79`), server binary `4c2a7c99be1c9b71…` built on the
box from that tree (`pro-single-day23/build.log`, `Finished release in 17.91s`, only `memra-server` recompiled
over the day-22 engine objects, `rc=0`), the Qwen3.8-27B NVFP4-Q5K MTP artifact, the boots as pre-registered.
One collector lock hold (`pro-single-day23/collector/stall-both-retry5/`, `CELL.jsonl` `"status":
"executed-not-qualified"`, `--validate` rc=0 in `stall-both-retry5.validate.log`) after FIVE bounded lock
retries of 120 s (`lock-retries.log`: attempts 0 to 4 `lock busy`; another lane held the box, never inspected,
never signalled); no compute app in the before or after snapshot (`stall/ev/compute-apps.{before,after}.csv`).
Regime from the collector's 250 ms sampler (`command.gpu.csv`, 849 rows): 32 to 51 C, 33.1 to 362.7 W under the
600 W limit, SM 180 to 2422 MHz. Marks (`stall/ev/marks.tsv`): OFF boot 04:08:58Z, ready in 6 s, demote arm
done 04:09:52Z, promote arm done 04:10:38Z; ON boot 04:10:40Z, ready in 14 s, demote arm done 04:11:42Z, promote
arm done 04:12:29Z. Harness digest `a5c9e761…` (`stall/ev/CELL.txt`), `stall_cell.py` unchanged. Every receipt
replayed: `STALL REPLAY: PASS (replay agrees with the harness's rule line)` four times (`stall/ev/replays.log`).

Verdict lines, verbatim (`stall/ev/{off,on}/{demote,promote}/receipt.json` `rule_line`; ms):

`STALL rule cell=stall-demote-off arm=demote n_per_order=5 pooled=10 idle_runs=10 idle_p50=13.5 idle_p95=14.7 idle_p99=14.9 idle_max=15.8 arm_runs=10 arm_p50=13.5 arm_p95=14.8 arm_p99=87.7 arm_max=132.1 stall_median=117.5 stall_min=78.5 stall_max=118.7 server_demote_ms=[38.1, 42.9, 42.9, 42.0, 42.4, 42.1, 41.5, 41.4, 41.7] server_promote_ms=[] intruder_prompt_tokens=[95, 99, 97, 98, 97, 99, 97, 97, 97, 97] tenant_text_identical=True errors=0`

`STALL rule cell=stall-promote-off arm=promote n_per_order=5 pooled=10 idle_runs=10 idle_p50=13.4 idle_p95=14.7 idle_p99=14.8 idle_max=16.8 arm_runs=10 arm_p50=13.4 arm_p95=14.7 arm_p99=16.6 arm_max=134.2 stall_median=85.2 stall_min=84.4 stall_max=120.8 server_demote_ms=[40.6, 42.1, 6.8, 6.3, 6.1, 6.1, 6.0, 6.1, 6.1, 6.0] server_promote_ms=[45.0, 46.5, 11.2, 10.6, 10.5, 10.4, 10.4, 10.5, 10.5, 10.3] intruder_prompt_tokens=[89, 86, 89, 86, 89, 86, 89, 86, 89, 86] tenant_text_identical=True errors=0`

`STALL rule cell=stall-demote-on arm=demote n_per_order=5 pooled=10 idle_runs=10 idle_p50=13.4 idle_p95=14.7 idle_p99=14.8 idle_max=15.0 arm_runs=10 arm_p50=13.4 arm_p95=14.8 arm_p99=130.9 arm_max=163.1 stall_median=149.5 stall_min=75.8 stall_max=149.7 server_demote_ms=[127.6, 132.6, 132.5, 132.2, 132.4, 132.0, 132.5, 132.2, 131.9] server_promote_ms=[] intruder_prompt_tokens=[95, 99, 97, 98, 97, 99, 97, 97, 97, 97] tenant_text_identical=True errors=0`

`STALL rule cell=stall-promote-on arm=promote n_per_order=5 pooled=10 idle_runs=10 idle_p50=13.4 idle_p95=14.7 idle_p99=14.8 idle_max=14.9 arm_runs=10 arm_p50=13.4 arm_p95=14.8 arm_p99=92.5 arm_max=131.0 stall_median=81.9 stall_min=81.4 stall_max=117.6 server_demote_ms=[207.7, 207.8, 172.6, 171.8, 172.3, 172.0, 172.4, 171.8, 172.5, 171.9] server_promote_ms=[61.3, 61.6, 26.4, 25.8, 25.9, 26.0, 25.9, 26.1, 25.9, 26.0] intruder_prompt_tokens=[89, 86, 89, 86, 89, 86, 89, 86, 89, 86] tenant_text_identical=True errors=0`

The pre-registered reading, as printed by `day23-stall-reading.py` (`pro-single-day23/stall/reading.log`),
verbatim:

    ON promote admissibility: submitted=10 published=10 h2d_receipts=10 cached_tokens=[64, 64, 64, 64, 64, 64, 64, 64, 64, 64] bad_lines=0 settled_synchronously_by_a_promote=0
    ON promote publish polls (recorded, no rule): [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] submission_to_completion_ms=[19.5, 19.6, 19.6, 19.5, 19.5, 19.5, 19.5, 19.7, 19.5, 19.5]
    admissible=True
    DAY23 STALL READING P1 promote ON stall against day 18 (81.9): at_or_under_day18 (stall_median=81.9 bounds <=81.9 | <=83.9)
    DAY23 STALL READING P2 promote ON server_promote_ms steady against day 18 (26.1): within_wait (steady_median=25.9 bound <=28.1; first_pair_max=61.6 first_pair_within_wait bound <=63.9)
    DAY23 STALL READING P3 promote ON against OFF, same window: on_at_off (on=81.9 off=85.2 bound on<=93.7)
    DAY23 STALL READING P4 promote OFF regime against day 16 (85.0): off_stable (off=85.2 band 76.5..93.5)
    DAY23 STALL READING P5 demote ON against day 18 (149.7): demote_half_unchanged (on=149.5 band 134.7..164.7)
    DAY23 STALL READING P6 demote OFF regime against day 16 (117.5): off_stable (off=117.5 band 105.8..129.3)
    DAY23 STALL READING summary: admissible=True P1=at_or_under_day18 P2=within_wait P3=on_at_off P4=off_stable P5=demote_half_unchanged P6=off_stable

**What the receipts say, read against the pre-registration and nothing else.** The claim held on both counts:
the promote ON stall median on the tree with the parked-only wait is `81.9`, the day-18 figure to the tenth
(`at_or_under_day18`, at the bound, not under it), and the promote's own submission-to-publication window is
`25.8` to `26.4` steady (median 25.9 against day 18's 26.1; first pair 61.3, 61.6 against 60.8, 61.9), inside the
2 ms the wait may add (`within_wait`). The copy's own submission-to-completion is `19.5` to `19.7` ms in every
`promote published off the tick` line, after `1 poll(s)`, as on day 18. So the wait that replaced the spin
neither delayed the publish nor stretched the tenant's tick by a measurable amount: the tick-top poll after a
bounded 2 ms command-channel wait lands where the spinning tick's poll landed. Same window, OFF against ON on
one tree (P3, the only same-window pair of the promote arm this lane has): ON `81.9` against OFF `85.2`,
`on_at_off`, 3.3 ms under OFF, the stretched tick `arm_max=131.0` against `134.2`. The OFF program read `85.2`
against day 16's `85.0` (`off_stable`) and `117.5` against `117.5` for the demote (`off_stable`), so the sitting's
regime is day 16's. The demote arm under the door read `149.5` against day 18's `149.7` and day 17's `149.6`
(`demote_half_unchanged`), with `server_demote_ms` `131.9` to `132.6` submission to publication as on day 17
(`127.9`, `132.3` to `134.0`); the wait fires only with a `Promoting` entry, and the demote arm has none.
What the cell does not say: nothing about the door's decision; nothing about the 5090 class (no stall cell exists
there); nothing about the 3 GB GLM entry shape; the `server_demote_ms` list in the promote arm (`171.8` to
`172.6`, the inline demote of each promote, submission to publication off the tick) is the day-18 figure and is
not the tenant's cost.

Same-box cross-sitting readings against A's day 16, 17 and 18 receipts (P1, P2, P4, P5, P6); P3 alone is a
same-window A/B. N=5 per arm per order, both orders, N=10 pooled, one card class, one host, one artifact,
`MEMRA_SERVE_SPEC=0`, 64-token entries (159.8 to 159.9 MB). `executed-not-qualified`.

## The door gates on main's tree after #627, local RTX 5090 Laptop GPU (review item 8, the lock freed)

Tree `91b0d4e08`, release `memra-server` `d4a3c3ffa78d1f76…` built locally under `systemd-run --user --scope -p
CPUQuota=1200% -p MemoryMax=28G` (`rtx5090-day23/build.log`, `Finished release in 18.45s` over the day-22
objects, `rc=0`), the Qwen3.5-9B NVFP4 MTP artifact, `MEMRA_HOSTGATE_CACHE_MB=64`, `day22-cell.sh` unchanged
(the day-22 driver; each gate under its own `flock -n /tmp/memra-5090.lock`, 15 x 120 s bounded retries, the
holder never signalled), ten cells in one sitting 04:02:08Z to 04:08:27Z (`rtx5090-day23/battery.log`). One
bounded lock retry: `failure-default-off` found the lock busy once (`lock-retries.txt`, `attempt 1: lock busy,
waiting 120 s`); its before snapshot carries another session's `target/release/memra-server` (10 MiB) beside the
`colbert-2/.venv/bin/python` co-tenant (1390 MiB) that sits in all twenty snapshots; neither is mine, neither was
touched. These are pass/fail gates with no timing claim, so the co-tenant changes nothing they assert; the
snapshots are part of the receipt. Receipts `rtx5090-day23/<cell>/` (`CELL.txt`, `gate.log`, `ev/`, compute apps
before and after, `verdict.txt`, `gate.exit`).

| Cell | Environment | Verdict line | ok / FAIL |
|---|---|---|---|
| fault-default | default (door ON by construction) | `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` | 65 / 0 |
| fault-plain | `MEMRA_SERVE_SPEC=0` | `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` | 65 / 0 |
| failure-default-off | default | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` | 15 / 0 |
| failure-default-on | default, `MEMRA_KV_HOST_CONTRACTS=1` | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` | 15 / 0 |
| failure-plain-off | `MEMRA_SERVE_SPEC=0` | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` | 15 / 0 |
| failure-plain-on | `MEMRA_SERVE_SPEC=0`, door ON | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` | 15 / 0 |
| identity-default-off | default | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` | 12 / 0 |
| identity-default-on | default, door ON | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` | 12 / 0 |
| identity-plain-off | `MEMRA_SERVE_SPEC=0` | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` | 12 / 0 |
| identity-plain-on | `MEMRA_SERVE_SPEC=0`, door ON | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` | 12 / 0 |

The fault gate's reject cell reads `items=18` (default) and `items=16` (plain) from the server's own r2 D2H
receipt with the floor line `ok: promote-reject: the entry carries at least two planes, so the reject is partial
(items=N >= 2)` in both, as on day 22. The whole-budget arm (`MEMRA_KV_HOST_TENANT_PCT=100`) was not run on this
card (unchanged from day 22: the 27B receipt stands for the arm; listed as missing item 9 in the review table).
This is the first run of the door gates on the RTX 5090 class on a tree carrying Move 1 whole with the lead's
three #627 fixes; day 22 ran only the fault gate there (on `98170f182`, before the fixes).

## The door gates on main's tree after #627, target card (review item 7; the lock was free after the stall cell)

Tree `91b0d4e08`, the same binary `4c2a7c99be1c9b71…`, `day22-box-run.sh` and `day22-cell.sh` reused unchanged
(twelve cells, `MEMRA_HOSTGATE_CACHE_MB=256`, one collector hold per cell, `--external-lock`), zero lock retries,
no compute app in any of the 24 before and after snapshots, `progress.log` 04:15:24Z to 04:23:15Z,
`BOX-BATTERY-DONE`. Receipts `pro-single-day23-gates/cells/<cell>/` and `pro-single-day23-gates/collector/<cell>/`
(twelve `CELL.jsonl`, each `"status": "executed-not-qualified"`).

| Cell | Environment | Verdict line | ok / FAIL |
|---|---|---|---|
| fault-default | default (door ON by construction) | `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` | 65 / 0 |
| fault-plain | `MEMRA_SERVE_SPEC=0` | `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` | 65 / 0 |
| failure-default-off | default | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` | 15 / 0 |
| failure-default-on | default, `MEMRA_KV_HOST_CONTRACTS=1` | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` | 15 / 0 |
| failure-plain-off | `MEMRA_SERVE_SPEC=0` | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` | 15 / 0 |
| failure-plain-on | `MEMRA_SERVE_SPEC=0`, door ON | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` | 15 / 0 |
| failure-default-pct100-off | `MEMRA_KV_HOST_TENANT_PCT=100` | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` | 14 / 0 |
| failure-default-pct100-on | `MEMRA_KV_HOST_TENANT_PCT=100`, door ON | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` | 14 / 0 |
| identity-default-off | default | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` | 12 / 0 |
| identity-default-on | default, door ON | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` | 12 / 0 |
| identity-plain-off | `MEMRA_SERVE_SPEC=0` | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` | 12 / 0 |
| identity-plain-on | `MEMRA_SERVE_SPEC=0`, door ON | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` | 12 / 0 |

The reject cell reads `items=34` (default) and `items=32` (plain) with the floor line; the whole-budget arm prints
`ok: pool-full refusal is LOUD and named (skip demote: entry X MB > host budget B MB)` in both door arms, 14 `ok`
(the tenant-reject count belongs to the share-cap arm). Day 22's twelve cells ran on `98170f182`, which carried
A's day 18 and #622 with my own worker.rs resolution; these twelve are the first on a tree whose `crates/` equal
main's after #627 (the lead's resolution plus the three revuto fixes). Verdicts identical to day 22's, cell for
cell.

## Task 2: the HOSTPREFIX door review table, final form

`HOSTPREFIX-DOOR.md`, "Review table for the decide-by (2026-10-05), final form (day 23)" replaces the day-18
table: A, the owed cells with tree, receipt path, verbatim verdict line and state (the stall cells of A's days
16, 17 and 18 and today's added as rows; today's is the one same-window OFF against ON pair of the promote arm);
B, the cost table (demote and promote stall medians on-tick and after Move 1, the server lines, the
cached-destination pair, the superseded write-combined pair, the hash micro-cell on both hosts, the ticket
lifecycle's share as arithmetic, the arena pair); C, the correctness table (identity, failure and fault gates,
both cards, both arms, the trees with a green receipt, today's rows on `91b0d4e08` for both cards); D, still
missing with the reason (1 the arena handoff scoped by ruling 28; 2 the DFlash tail slice, pre-registered, its
identity input recorded day 20, no gate can boot a drafter; 3 verify digest v3; 5 the RTX 5090 class pair; 6 the
promote-side census question and the day-16 write-combined contradiction; 9 the whole-budget arm on the 5090;
items 7 and 8, the door gates on the post-#627 tree on each card, resolved today); E, the decision question in
one paragraph, not answered.

## Hygiene on the final tree

`cargo fmt --all -- --check` rc=0 on the merge tree (`day23-cpu/fmt-merge.log`; no Rust file changed after it),
`cargo test -p memra-server --lib` `788 passed; 0 failed; 14 ignored` (`day23-cpu/server-tests-merge.log`, the
lead's three round censuses included); `tools/check-flags.sh` no uncovered runtime names (no new `MEMRA_*` read:
the drivers set existing names); `tools/check-conflict-markers.sh` OK; `python3 tools/check-public-boundary.py
check` 0 new (the receipts carry `127.0.0.1` only); `git diff --check` clean; shellcheck silent on the two
drivers (SC1091 info on the sourced port guard only); `py_compile` on the reading script.

## Pushes

`d2b8efbf2` (the merge), `91b0d4e08` (drivers and this file's pre-registration, before the run), `357d98f0e`
(local RTX 5090 receipts), `2ce5a64f7` (target-card stall receipts), then the gate receipts and the docs, each in
`MEMRA_RELEASE_QUALIFICATION_MODE=development` (printed `UNQUALIFIED DEVELOPMENT ... no GPU qualification
claimed`, logged in the clone's `.git/memra-gate-skips.log`). Not merged into main, no PR opened.

## Left as it was, and cleanup

Not touched: `/root/artifacts`, `/root/memra-spill`, other lanes' worktrees or processes (the box lock was held
by another lane for five retries before the stall cell; the local card carried another session's server for one
retry and the `colbert-2` co-tenant throughout; seen only in the drivers' snapshots). Cleaned: `/root/day23.bundle`,
`/tmp/spill-c-day23` (its logs banked under `day23-cpu/`), no server of mine on either card, both locks free at
close; `/root/wt-c` left checked out at `91b0d4e08`, clean. Open for the lead: nothing new; the whole-budget arm's
copy-then-refuse cost under the door (day 22) and the `docs/TESTING.md` pointer question stand.
