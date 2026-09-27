# Session C day 26: the 5090 door gates on the slice-2 tree, the restore route's engagement on this card, the draft-bearing restore census

Lane `lane/spill-c-20260919`, checkout `wt-spill-c`. Every push today in the announced
`MEMRA_RELEASE_QUALIFICATION_MODE=development` mode (the #589 hook refuses the tree `UNQUALIFIED` because the
content-bound census sees the engine files integ37 brought in; the hook prints `UNQUALIFIED DEVELOPMENT:
refs/heads/lane/spill-c-20260919 at <sha>; no GPU qualification claimed` and records the skip in the clone's
`.git/memra-gate-skips.log`). Every cell below is `executed-not-qualified` development evidence; nothing here is a
qualification claim. No commit on main, no PR. No engine change of my own today.

## Merge (first action)

#638 (integ37) was OPEN, so `origin/lane/spill-integ37-20260922` `a3eae3a77` (A's day 21, Move 2 slice 2: the hit
restore off the tick behind rule 3's reader fence, `D2dRestore`, the `Restoring` request; the lead's integ37 record;
main `5df11152f`) merged `--no-ff` as `b74269af5`, clean, no conflict; `tools/check-conflict-markers.sh` OK;
`git diff origin/lane/spill-integ37-20260922 -- crates/ tools/` EMPTY. Pushed.

## Task 1, the 5090 door gates on the slice-2 tree (owed by A)

Tree `b74269af5`, release `memra-server` built locally under `systemd-run --user --scope -p CPUQuota=1200% -p
MemoryMax=28G` (`day26-cpu/build-local.log`, `rc=0`; digest in `rtx5090-day26/binary.sha256`). The sixteen cells of
day 24 (`day26-cell.sh`, derived from `day24-cell.sh`: the engine unit cell widened to `d2d_capture_*` and the new
`d2d_restore_*`, and a restore-route census per cell into `restore-lines.txt`), the 9B for the host-tier gates and the
hit gate, the 27B for the twin gate, `MEMRA_HOSTGATE_CACHE_MB=64`; every gate takes the canonical lock
`/tmp/memra-5090.lock` itself, compute apps and driver free sampled before and after each cell.

**Attempt 1, twelve cells failed at boot, kept as `rtx5090-day26/attempt1-oom/`.** The first cell waited two lock
retries (07:36 to 07:40Z, `fault-default/lock-retries.txt`), took the lock, and its server died at model load:
`[server] FATAL: worker init failed: load gate: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")`, verbatim
in `attempt1-oom/fault-default/gate.log`. The concurrent GPU state at failure, from the cell's own after-snapshot
(`compute-apps.after.csv`): `1200326, /tmp/tmp.pTuTeg4NH9/hold, 22176 MiB` with `memory.free` 1786 MiB of 24463; the
before-snapshot (07:38Z) had read 23970 MiB free and no compute app. A foreign process took 22 GB of the card
between my gate's lock acquisition and its boot, outside the canonical lock; it was not inspected beyond
nvidia-smi's listing and not signalled. The eleven following cells failed the same way in about two seconds each
(same line, same holder in every after-snapshot) until I stopped my own battery by pid (`kill` on the pids I
started, no pattern); by then the holder had gone and another session's `kernel-check` (1604 MiB) was on the card.
No verdict comes from attempt 1: every cell there is a boot failure with its cause quoted. Fix to the driver,
not to any gate or claim: `day26-local-battery.sh` now waits, bounded (15 x 120 s), for no compute app and at least
20000 MiB free before every cell, logging each wait with the compute-apps snapshot.

**Attempt 2, the sixteen cells, `rtx5090-day26/`.** Binary `e5ec91f45a80d5ca...` (`binary.sha256`), tree `b74269af5`.
The battery waited five times (07:41:46Z to 07:51:47Z, `battery.log`: another session's `kernel-check` 1604 MiB, then its
`graph-warmup-stress` 290 MiB, then two `memra-server` boots of 4066 and 10948 MiB; none inspected or signalled), then
ran 07:51:47Z to 08:01:21Z with no lock retry; no compute app in any cell's before or after snapshot; card 63 to 71 C
at the cells' starts. Verbatim, one row per cell (`verdict.txt`; ok and FAIL counts from `gate.log`):

| Cell | Environment | Verdict line | ok / FAIL | Restore route (`restore-lines.txt`) |
|---|---|---|---|---|
| fault-default | default (door ON by construction) | `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` | 67 / 0 | 0 lines; accounting `receipt seq=1 expected 1 + 0 capture ticket(s) submitted before it = 1`, `receipt seq=2 expected 2 + 0 capture ticket(s) submitted before it = 2` |
| fault-plain | `MEMRA_SERVE_SPEC=0` | `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` | 67 / 0 | 8 lines, ENGAGED (below); accounting `receipt seq=3 expected 1 + 2 capture ticket(s) submitted before it = 3`, `receipt seq=4 expected 2 + 2 capture ticket(s) submitted before it = 4` |
| failure-default-off | default | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` | 15 / 0 | 0 |
| failure-default-on | default, `MEMRA_KV_HOST_CONTRACTS=1` | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` | 15 / 0 | 0 |
| failure-plain-off | `MEMRA_SERVE_SPEC=0` | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` | 15 / 0 | 0 |
| failure-plain-on | `MEMRA_SERVE_SPEC=0`, door ON | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` | 15 / 0 | 0 |
| identity-default-off | default | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` | 12 / 0 | 0 |
| identity-default-on | default, door ON | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` | 12 / 0 | 0 (the entries are draft-bearing: refused by name) |
| identity-plain-off | `MEMRA_SERVE_SPEC=0` | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` | 12 / 0 | 0 (door OFF) |
| identity-plain-on | `MEMRA_SERVE_SPEC=0`, door ON | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` | 12 / 0 | 4 lines, ENGAGED (below) |
| hit-off | door OFF (`spec-on-cache-hit-gate.sh qwen`, 9B) | `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` | 61 / 0 | 0 |
| hit-on | door ON | `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` | 61 / 0 | 0 (the gate arms no host tier: below) |
| twin27-off | door OFF, the 27B | `PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=736755712 turns=8 cold_turns_after_1=0 cached_ok=7/7 lines_ok=8/8 evictions=9 cohort_evictions=3 self_evictions=0 refused_or_skipped=0 effective_free_ok=8/8 identity_ok=8/8 grid_ok=21/21 grid=32 off_grid_calls=0 V1=ok V2=ok V3=ok V4=ok V5=ok V6=ok -> PASS` | PASS (no `REFUSED` line) | 0 |
| twin27-on | door ON, the 27B | the identical line, `-> PASS` | PASS (no `REFUSED` line) | 0 |
| unit-server (`option_b_*`, `option_c_*`) | the door's unwinds on the card | `test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 809 filtered out; finished in 0.29s` | 8 / 0 | n/a |
| unit-engine (`d2d_capture_*`, `d2d_restore_*`) | the two D2D classes on the card | `test tier_transfer::tests::d2d_capture_lands_on_the_copy_stream_and_publishes_only_after_its_event ... ok`, `test tier_transfer::tests::d2d_restore_lands_on_the_copy_stream_and_is_ready_only_after_the_installed_wait ... ok`, `test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 554 filtered out; finished in 0.55s` | 2 / 0 | n/a |

**Where the restore route engaged on this card.** In two arms, both plain, both with the host tier armed and the
door ON; zero `restore refused`, `restore dropped` or `RESTORE OFF-TICK DISABLED` lines in any cell. Verbatim:

- `identity-plain-on/ev/host-on-server.log` (the promoted E_A, 64 tokens, hit by r3 and r4 at `hit: 64 of 102 prompt
  tokens from cache`): `[prefix-cache] restore submitted off the tick: 64 tokens, 16 planes (53.6MB), ticket seq=6 on
  the contracts door's copy stream; recurrent state copied on the owner stream; request parked`, then `[prefix-cache]
  restore landed off the tick: 64 tokens (53.6MB) complete after 1 poll(s), 0.0ms from submission to completion,
  0.1ms to re-admission (tick-top poll)`; and `... ticket seq=7 ...`, `... complete after 1 poll(s), 28.4ms from
  submission to completion, 28.5ms to re-admission (tick-top poll)`. The identity clause over those two restores is
  the gate's r3 ON == OFF and r4 == r3 byte identity, `teeth=0`, 12 ok: the one-program proof of the slice on the
  RTX 5090 class, as the plain identity arm was on the target card (A day 21).
- `fault-plain/ev/promote-{presubmit,postpublish,readyview,reject}-server.log`, one restore each, all `64 tokens, 16
  planes (53.6MB)`, tickets `seq=8`, `seq=9`, `seq=9`, `seq=9`, landed `28.8ms`, `29.0ms`, `28.4ms`, `28.8ms` from
  submission to completion, each `after 1 poll(s)` at a `tick-top poll`; the four promote cells' verdicts are the
  gate's 67 ok. (The promote cells' hit follows the promote; the two demote cells publish and evict and hit nothing
  whole.)

The 53.6 MB is the entry's byte count as the line prints it (the 9B's 16 KV planes plus its recurrent state; the
planes' share at 64 tokens is small, as the 27B's 1.9 MB of 158.8 MB was), and the 28 to 29 ms is submission to the
OBSERVING poll, the idle-wait period on this card, not the copy (the 0.0 ms line is a poll that ran right behind the
submission). Not engaged: the default (spec) arms, whose entries are draft-bearing and refused by name; the hit gate
in both arms, which boots without `MEMRA_KV_HOST_MB` (zero `[prefix-host]` lines in its four server logs) so the route
is off at `hpx.armed()` before any class check; the failure gate, whose shapes evict and never hit an entry whole with
the tier armed (0 lines in all four arms); the twin gate (door ON, no host tier). The hit gate's class census today
matches the target card's exactly: spec-on boot 11 `insert (spec-boundary)` + 1 `insert (seed)`, 13 hits (`64 of 106`
x6, `64 of 119` x3, `96 of 119` x2, `96 of 135`, `128 of 128`); spec-off twin boot 2 `insert (seed)`, 3 hits.

## Task 2, the restore-exercising identity cell: engaged in task 1, cited, skipped

The pre-registered condition for a new cell was "no engagement on the 5090 in task 1". The route engaged in
`identity-plain-on` (two restores, seq=6 and seq=7, the lines above) and in `fault-plain` (four), on the 64-token
promoted entry that the identity gate's r3 and r4 hit whole (`lookup`'s floor is `PREFIX_CACHE_MIN_TOKENS=64` on
every model; the entry is exactly at it; the 9B's plane count, 16 against the 27B's 32, is not in the route's
predicate). No new cell was run; the arithmetic is not owed because the smallest shape that engages the route is the
identity gate's own plain arm on both cards.

## Task 3, the draft-bearing restore census

`HOSTPREFIX-DOOR.md` section D item 11 (no code): the route's six silent gates before its class check, the seven
classes it refuses by name (TP shards, latent planes, the MTP draft plane, the DFlash tail, a partial entry, empty
boundary logits, no KV plane), what a draft plane's off-tick restore would need read from the OFF program it must
equal (`spec_session_from_restored_deferred`: the `MtpScratch` allocated at the probe and owned by the `Restoring`
state, a borrowed destination view of `pos x k_tok_bytes` and `pos x v_tok_bytes`, the source under the trunk's pin,
the producer fence on the owner stream, rule 3's wait installed before the deferred prime's first draft-head read,
`spec_restore_refusal` decided at the probe so a plain-serving request never gets a draft plane, the geometry checks
moved to the probe, slice 3's receipt term over `Role::Draft`: `items=34` on the 27B, `items=18` on the 9B), the
fraction (11 of 14 hit-gate entries and 12 of 16 hits draft-bearing, identical on both rigs), and the finding that
the hit gate arms no host tier, so its identity clause covers the route on no boot as the gate stands.

## Hygiene on the final tree

`cargo fmt --all -- --check` rc=0; `git diff --check` clean; `tools/check-flags.sh` no uncovered runtime names (no
new `MEMRA_*` read); `tools/check-conflict-markers.sh` OK; `python3 tools/check-public-boundary.py check` 0 new;
shellcheck `-S warning` clean on the two drivers; no em dash in any file written today.

## Pushes

`b74269af5` (the merge), `e91b44e7f` (drivers, item 11, attempt 1), `ee76d09e9` (the sixteen cells), then the closing
commit (this file, STATE, INDEX, the door review row), each in `MEMRA_RELEASE_QUALIFICATION_MODE=development`
(printed `UNQUALIFIED DEVELOPMENT ... no GPU qualification claimed`, logged in the clone's `.git/memra-gate-skips.log`).
Not merged into main, no PR opened.

## Left as it was, and cleanup

BOX3 not touched today (no cell needed the target card: every owed cell was a 5090 cell). Local RTX 5090: every gate
took the canonical lock and released it; the only processes I stopped were my own battery, cell and gate shells by
pid; compute apps empty at the last cell's after-snapshot. `/tmp/day26-battery.pid` removed. No bundle, no `/tmp`
scratch left. Other sessions' processes on the card (the 22 GB holder, `kernel-check`, `graph-warmup-stress`, two
`memra-server` boots) were seen only in nvidia-smi listings, never inspected further or signalled.

## Budget

About 2.1 agent-hours against 4: reading and the merge 0.3, the drivers and the census 0.5, the OOM attempt, the
waits and the battery 0.7, the records and checks 0.6. Blockers: none open. For lane A: the 5090 door gates on the
slice-2 tree are green in every arm and the route engaged on this card in the plain identity and fault arms.
