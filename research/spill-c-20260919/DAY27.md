# Session C day 27: the hit gate's door arm armed and asserted, run OFF and ON on both cards; the review table corrected

Lane `lane/spill-c-20260919`, checkout `wt-spill-c`. Every push today in the announced
`MEMRA_RELEASE_QUALIFICATION_MODE=development` mode (the #589 hook refuses the tree `UNQUALIFIED` because the
content-bound census sees engine files in the range; the hook prints `UNQUALIFIED DEVELOPMENT:
refs/heads/lane/spill-c-20260919 at <sha>; no GPU qualification claimed` and records the skip in the clone's
`.git/memra-gate-skips.log`). Every cell below is `executed-not-qualified` development evidence; nothing here is a
qualification claim. No commit on main, no PR. No engine change today: two changes to one gate script.

## Merge (first action)

`origin/main` `58b814abe` (#638 integ37 merged: A's Move 2 slices 1 and 2 with the lead's revuto fixes) merged `--no-ff`
as `2f8560951`, clean, no conflict (7 files, `crates/memra-server/src/worker.rs` among them: the #638 round-2 fixes);
`tools/check-conflict-markers.sh` OK. Pushed.

## Task 1, the hit gate's door arm (`5c8c72447`, `tools/spec-on-cache-hit-gate.sh`)

**What was wrong.** The door batteries run `spec-on-cache-hit-gate.sh qwen` twice, with `MEMRA_KV_HOST_CONTRACTS` unset
(door OFF) and `=1` (door ON). The gate never exported `MEMRA_KV_HOST_MB`, so under the door the server built no
program identity and said so on its own line, verbatim in every ON-arm log on both lanes:
`[kv-host-contracts] MEMRA_KV_HOST_CONTRACTS=1 with no host tier on this boot (MEMRA_KV_HOST_MB=0): nothing to route,
no program identity built`. `hpx.armed()` (`budget > 0 && !disabled`) was false before any entry-class check, the
capture and restore routes need `hpx.tier` (built only when `kv_host_contracts && hpx.budget > 0`), and the ON arm
was the OFF arm under another name. Census of the banked logs (the day-26 finding, now enumerated): zero
`[prefix-host]` lines and the `[kv-host-contracts]` line in lane A's `pro-single-day{17,18,18-run1,19,20,21}/box/gates/
hitgate-on/` and `rtx5090-day{17,17-attempt1,18}/hitgate-on/` (A's day-16 runs had no door arm), and in this lane's
`rtx5090-day24/hit-on` and `rtx5090-day26/hit-on`. Every "hit gate ALL GREEN OFF and ON" taken from those runs covered
the tick program in both arms.

**The change.** With `MEMRA_KV_HOST_CONTRACTS=1` in its environment the gate now (a) exports the identity gate's host
budget to BOTH boots, spec-on and the spec-off twin: `MEMRA_KV_HOST_MB=8192` (`kv-host-spill-identity-gate.sh`'s
`MEMRA_HOSTGATE_HOST_MB` default; an exported `MEMRA_KV_HOST_MB` is respected); (b) asserts per boot the tier's arming
line `[prefix-host] on: budget`, the door's `[prefix-host] contracts door ON`, and no latch line (`TIER DISABLED`,
`CAPTURE OFF-TICK DISABLED`, `RESTORE OFF-TICK DISABLED`); (c) asserts across the two boots at least one route
submission, `(capture|restore|demote|promote) submitted off the tick` (pre-registered reason: the spec-off twin's
`insert (seed)` entries take the capture route by construction, the contract fault gate's `1 + 2 capture ticket(s)`
accounting on both cards; the restore route on the plain hits is censused, not asserted); (d) prints, in both arms, an
entry-class census per boot: `insert (spec-boundary)` (draft-bearing) against `insert (seed)` (plain) counts, the hit
shapes, the class of the entries in the identity clause's own namespaces (the default namespace for r1..r3 and ns
"grow" for g1..g4), and the door-line counts; (e) quotes the arming line, the door line and the first route line
verbatim under the assertions. The OFF arm is unchanged (no `MEMRA_KV_HOST_MB`, no assertion, census only). The
identity clause (r1, r2, r3, g1, g2 spec-on text == spec-off text) is unchanged in both arms. The door arm is defined
for the qwen arm only (`REFUSED` with exit 2 on gemma: a gemma tower is a boot refusal under the door). No new
`MEMRA_*` read (`MEMRA_KV_HOST_CONTRACTS` and `MEMRA_KV_HOST_MB` are FLAGS rows; `tools/check-flags.sh` clean). Dry
run before committing: the day-26 `hit-on` log reads two `FAIL` under `door_assert` (the arm that used to read ALL
GREEN), the day-26 `identity-plain-on` host log reads three `ok` with the capture route as its first route line.

**Not added, stated.** The gate has no `--external-lock`; it takes the canonical lock itself with `flock -w 300` per boot
inside `boot()`, so under the collector's hold it would block. Lane A day 21 and C day 26 ran it under its own
`flock`; today the same, on both cards, with my driver recording compute apps and the card before and after.

## Task 1, the runs (tree `b1e9c75b6`, the gate above plus the drivers; `day27-cell.sh`, `day27-local-battery.sh`,
## `day27-box-run.sh`)

**Local RTX 5090 Laptop GPU, the 9B** (`rtx5090-day27/`; release `memra-server` built under `systemd-run --user --scope
-p CPUQuota=1200% -p MemoryMax=28G`, `day27-cpu/build-local.log` `rc=0`, digest `6020ff1fd108408f...`; lock
`/tmp/memra-5090.lock`). The battery waited four times (08:19 to 08:27Z, `battery.log`: other sessions' `kernel-check`
1604 MiB, `decode-batch-gate` 17634 MiB, `decode-dc-gate` 7330 MiB, a `memra-server` 290 MiB; none inspected or
signalled), then ran with no compute app in any before or after snapshot, 56 to 63 C:

| Cell | Environment | Verdict line | ok / FAIL | Door lines | Route lines |
|---|---|---|---|---|---|
| hit-off | door OFF | `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` | 61 / 0 | 0 (`armed=0 door_on=0` on both boots) | 0 |
| hit-on | `MEMRA_KV_HOST_CONTRACTS=1` (the gate exports `MEMRA_KV_HOST_MB=8192`) | `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` | 68 / 0 | 18 | 8 restore lines, 7 submissions in all |

**Target card, one RTX PRO 6000 Blackwell at 600 W, the 27B** (`pro-single-day27/`; the tree shipped as a bundle,
fetched into `/root/wt-c`, checked out detached; release build on the box `build.log` `rc=0` in 3m 15s, digest
`39c6f8f2a17d9dbc...`; lock `/tmp/memra-gpu.lock`, free at the start, no retry; card idle before and after both cells,
`0 MiB` used, 33 to 39 C; the runner under `nohup`):

| Cell | Environment | Verdict line | ok / FAIL | Door lines | Route lines |
|---|---|---|---|---|---|
| hit-off | door OFF | `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` | 61 / 0 | 0 | 0 |
| hit-on | door ON | `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` | 68 / 0 | 18 | 8 restore lines, 7 submissions in all |

**The ON arm's engagement, verbatim (the 27B; the 9B prints the same lines at `16 planes (53.6MB)`).** Both boots:
`[prefix-host] on: budget 8590MB pinned cacheable host RAM (MEMRA_KV_HOST_MB, startup budget policy), plain byte-LRU,
demote on device capacity eviction, promote on exact-prefix probe; verify=off (MEMRA_KV_HOST_VERIFY); tenant share cap
50% = 4295MB (MEMRA_KV_HOST_TENANT_PCT)` and `[prefix-host] contracts door ON (MEMRA_KV_HOST_CONTRACTS=1): 1 model
program identities, tenant salt per pool namespace, server governor ledger pinned/pageable 17180MB device 6442MB
in-flight 390; host tier armed; ...` (`in-flight 198` on the 9B). Spec-off twin: `[prefix-cache] capture submitted off
the tick (seed): 64 tokens, 32 planes (158.8MB) on the contracts door's copy stream; recurrent state cloned at the
boundary on the owner stream` x2 (its two seeds, published `after 1 poll(s), 0.1ms` and `after 2 poll(s), 87.5ms`), and
`[prefix-cache] restore submitted off the tick: 64 tokens, 32 planes (158.8MB), ticket seq=2 on the contracts door's
copy stream; recurrent state copied on the owner stream; request parked` then `seq=3` and `seq=5` (its r2, r3 and g2
hits, `hit: 64 of 106` and `64 of 119` x2), each `restore landed off the tick: 64 tokens (158.8MB) complete after 1
poll(s), 2.1ms from submission to completion, 2.1ms to re-admission (tick-top poll)` (2.1 to 2.2 ms on both cards).
Spec-on boot: one capture (the `samp-noplane` seed, the boot's one plain entry) and one restore (`seq=2`, the np hit on
it). Gate lines: `ok: door arm: the host tier is armed on the spec-on boot`, `... contracts door is ON ...`, `... no latch
line ...` (and the same three for the spec-off boot), `ok: door arm: 7 route submission(s) across the two boots (capture,
restore, demote or promote off the tick)`. Zero `refused (contracts door)`, `restore refused`, `dropped` or `DISABLED`
lines in any log.

**What the identity clause covered, from the census (identical on both cards).** Spec-on boot: `11 draft-bearing
(insert (spec-boundary)), 1 plain (insert (seed))`, hits `128 of 128` x1, `64 of 106` x6, `64 of 119` x3, `96 of 119`
x2, `96 of 135` x1; `identity clause namespaces (default, "grow") on the spec-on side: 4 draft-bearing, 0 plain -> its
hits restored draft-bearing entries`. Spec-off twin: `0 draft-bearing, 2 plain`, hits `64 of 106` x1, `64 of 119` x2;
`... on the spec-off side: 0 draft-bearing, 2 plain -> its hits restored plain entries`. So in the ON arm the clause's
r2, r3 and g2 compare the spec side's tick-program hits on draft-bearing entries (the route refuses `e.draft` by name)
against the plain side's hits restored THROUGH the door's off-tick route, byte for byte, and the clause held on both
cards. This is the first hit-gate receipt on any card where the identity clause covers the route; it covers it for the
plain hits. The draft-bearing restore stays owed (item 11), and the spec side's `ALL GREEN` says nothing about it.

**Verdict on the door.** Not red: the ON arm under an armed tier read `ALL GREEN` on both cards; no finding against the
door from this gate today. The clause was not touched.

## Task 2, the review table (`HOSTPREFIX-DOOR.md`)

The hit gate appears in the review table in one row (the day-26 sixteen-cell row, section C) and in item 11; section A
had no hit-gate row. Done: the day-26 row carries a DAY 27 NOTE naming the unarmed condition with the server's own
`[kv-host-contracts]` line (the two hit cells stay as receipts); two new section-C rows carry today's armed runs per
card with the engagement lines; section A gains an owed-cell row for the armed hit gate, whose state column lists every
earlier unarmed ON arm (A days 17 to 21 on the target card and 5090 days 17 and 18; C 5090 days 24 and 26); item 11
gains its day-27 sentence. Lane A's own day records and the lead's integ37 record carry their "hit OFF and ON ALL
GREEN" lines unchanged; they are those lanes' files, and this record plus the review table say what those runs covered.

## Gate hygiene found on the shared box (`c1454a5a4`)

While the gate ran on the target card, lane A's day-22 `memra-server` appeared on the card right after my cells. The
gate's `stop()` was a blanket `pkill -x memra-server`, and `trap stop EXIT` runs it a second time after the last
boot's `flock` has been released, so a server another lane boots in that window could be killed. `stop()` now kills
only the child of this gate's own `flock` wrapper (`pkill -x -P "$SERVER_PID" memra-server`; `flock` forks, the child
execs `env` which execs the binary) and does nothing when no boot of ours is outstanding. No assertion or boot
environment changed. Whether A's server was touched I cannot say and did not look: it was seen only in nvidia-smi's
listing (16490 then 21706 MiB) after my last cell; the after-snapshot of my `hit-on` shows `0 MiB` used, so no server of
any lane was on the card when my trap fired at 08:23:07Z. Re-runs on the fixed tree `c1454a5a4`:

| Card | Cell | Verdict line | ok / FAIL | Receipts |
|---|---|---|---|---|
| RTX 5090, 9B | hit-off | `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` | 61 / 0 | `rtx5090-day27-stopfix/hit-off` (08:30:54 to 08:31:18Z, card idle before and after) |
| RTX 5090, 9B | hit-on | `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` | 68 / 0 | `rtx5090-day27-stopfix/hit-on` (18 door lines, 7 submissions; no `port already serving` and no `server died` line: the second boot found the port free) |
| RTX PRO 6000, 27B | hit-off | `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` | 61 / 0 | `pro-single-day27-stopfix/cells/hit-off` (two bounded lock waits, 08:30:57 and 08:32:57Z, behind lane A's hold, the holder never inspected or signalled; ran 08:34:57 to 08:35:34Z, card `0 MiB` before and after, 34 to 40 C) |
| RTX PRO 6000, 27B | hit-on | `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` | 68 / 0 | `pro-single-day27-stopfix/cells/hit-on` (08:35:35 to 08:36:26Z; 18 door lines, 7 submissions; no `port already serving` and no `server died` line) |

## Hygiene on the final tree

`cargo fmt --all -- --check` rc=0; `git diff --check` clean; `tools/check-flags.sh` no uncovered runtime names;
`tools/check-conflict-markers.sh` OK; `python3 tools/check-public-boundary.py check` 0 new; shellcheck `-S warning` on
the gate and the three drivers: one pre-existing SC2034 (`tries` in `ongrid_prompt`, not today's); no em dash in any
line added today (the gate header, `docs/TESTING.md` and `research/INDEX.md` carry older ones).

## Pushes

`2f8560951` (the merge), `5c8c72447` (the door arm), `b1e9c75b6` (drivers), `770844954` (the four cells on both cards),
`c1454a5a4` (stop() hardening), then the closing commits (this file, STATE, INDEX, the review table, `docs/TESTING.md`,
the stop-fix receipts), each in `MEMRA_RELEASE_QUALIFICATION_MODE=development` (printed `UNQUALIFIED DEVELOPMENT ...
no GPU qualification claimed`, logged in the clone's `.git/memra-gate-skips.log`). Not merged into main, no PR opened.

## Left as it was, and cleanup

BOX3 reached through the existing control socket only (`ssh -O check` first, `Master running`); `/root/wt-c` (mine)
left detached at `c1454a5a4`, clean; the two bundles removed on both ends; receipts under `/root/spill-receipts/day27` and
`day27-stopfix` (copied here); `/root/artifacts`, `/root/memra-spill` and other lanes' worktrees and processes not
touched; lane A's server seen in nvidia-smi's listing only. Local: every gate took the canonical lock and released it;
`/tmp/day27-battery.pid` removed; no `/tmp` scratch left.

## Budget

About 2.6 agent-hours against 4: reading and the merge 0.4, the gate change and its dry run 0.5, the drivers, the
bundle and the two builds 0.4, the runs and their reads 0.5, the review table, TESTING, the records and the checks 0.6,
the stop() finding and its re-runs 0.2. Blockers: none open (the box stop-fix re-run waited two bounded rounds behind lane
A's hold and landed).
