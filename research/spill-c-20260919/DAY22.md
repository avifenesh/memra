# Session C day 22: the fault gate with its floor, the failure gate's whole-budget arm run for the first time, and the three host-tier gates on the tree that carries Move 1 whole (A's demote and promote halves under the door)

Lane `lane/spill-c-20260919`, checkout `wt-spill-c`. Every push today in the announced
`MEMRA_RELEASE_QUALIFICATION_MODE=development` mode (the #589 hook refuses the tree `UNQUALIFIED` because the
content-bound census sees the engine files main and lane A brought in; the hook prints `UNQUALIFIED DEVELOPMENT:
refs/heads/lane/spill-c-20260919 at <sha>; no GPU qualification claimed` and records the skip in the clone's
`.git/memra-gate-skips.log`). Every cell below is `executed-not-qualified` development evidence, pass/fail, no
timing claim. No commit on main, no PR. No engine change of my own today: the one source edit is the merge
resolution below, which keeps both sides' behavior. A first launch of this day was stopped by the lead before it
changed anything; the worktree was clean at `304e8235c` when this one started.

## Merges (first action)

- `origin/main` `3df055601` (#622 integ30, A's day 17 with the lead's two revuto fixes; #626 integ31, my day 21
  with the lead's floor) merged clean as the first parent step (no conflict).
- `origin/lane/spill-a-20260919` `614c53f71` (A day 18: the promote half of Move 1 under the door) merged as
  `cf45be796`. Two conflicts:
  - `crates/memra-server/src/worker.rs`, two hunks, both real: main's `089cb37e4` (revuto round 2 on #622:
    re-check `armed()` after the settle-first at `host_demote_prefix_ref` and `host_promote_prefix_hit`, with a
    typed refusal line each) met A's day 18 (a `Promoting` settle added at both sites; the promote hook's
    settles moved behind the candidate check and the memo, with a re-lookup after them). Resolved keeping both:
    at `host_demote_prefix_ref` the demote settle, then A's `host_promote_settle_contract`, then main's
    `armed()` re-check (either settle can latch the tier off); at `host_promote_prefix_hit` A's order (candidate,
    memo, demote settle, promote settle) then main's `armed()` re-check with its line, then A's re-lookup. Main's
    line text is unchanged; A's settle-first census (`host_promote_candidate(` before the settle, the settle
    before `host_promote_prepare(`) holds. Gates: `cargo fmt --all -- --check` rc=0, `cargo test -p memra-server
    --lib` `786 passed; 0 failed; 14 ignored` (`day22-cpu/`).
  - `research/INDEX.md`: the inherited recursive-merge marker block on one side, A's `spill-a-20260919/day18`
    row on the other; the markers dropped, the row kept; `tools/check-conflict-markers.sh` OK before the commit.

The tree under every cell today is `98170f182` (the two merges plus the day-22 drivers). It carries Move 1 whole:
the D2H demote off the tick (A day 17), the H2D promote off the tick with the parked request (A day 18), the
lead's fail-closed settle arm and the two round-2 fixes, and my two day-21 gate fixes with the lead's floor.

## Drivers

`day22-cell.sh` is `day21-cell.sh` plus one named environment: a `-pct100-` cell sets
`MEMRA_KV_HOST_TENANT_PCT=100` for the gate and the server it boots (the gate's `boot` inherits the environment).
`day22-box-run.sh` takes its cell list as arguments so a partial re-run names its cells; the default list is the
twelve cells below. Both through the collector on the target card (`tools/tier-battery.py --rig pro-single
--timeout 3600 --external-lock`, `/tmp/memra-gpu.lock`, 75 x 120 s bounded retries), the gate's own `flock -n` on
the canonical `/tmp/memra-5090.lock` locally (15 x 120 s bounded retries; the holder is never signalled).

## Task 1 and 2: the target card (one RTX PRO 6000 Blackwell Server Edition, 600 W limit, collector rig `pro-single`)

Tree `98170f182`, server binary `61cc239845f431e9…` built on the box from that tree (`pro-single-day22/build.log`,
`Finished release in 3m 08s`, `rc=0`), the 27B NVFP4-Q5K MTP artifact, `MEMRA_HOSTGATE_CACHE_MB=256`. Receipts:
`pro-single-day22/cells/<cell>/` (`CELL.txt`, `gate.log`, `ev/`, compute-apps before and after, `verdict.txt`,
`gate.exit`) and the collector's `pro-single-day22/collector/<cell>/` (`CELL.jsonl` `"status":
"executed-not-qualified"`). The box was free when the battery started: zero lock retries, no compute app in any
of the 24 before/after snapshots. Twelve cells, `progress.log` 02:55:58Z to 03:03:41Z, `BOX-BATTERY-DONE`.

| Cell | Environment | Verdict line | ok / FAIL |
|---|---|---|---|
| fault-default | default (door ON by construction) | `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` | 65 / 0 |
| fault-plain | `MEMRA_SERVE_SPEC=0` (door ON by construction) | `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` | 65 / 0 |
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

**The fault gate with the floor (task 1).** The reject cell reads, verbatim, in the default environment:

    promote-reject: the entry's plane count from the server's r2 D2H receipt: items=34
    ok: promote-reject: the injected refusal's `1 of M items` M equals that receipt's items=N
    ok: promote-reject: the entry carries at least two planes, so the reject is partial (items=N >= 2)

and in the plain environment the same three lines with `items=32`. The gate's `ok:` count is 65 (day 21 read 64:
the floor is the one added check). The refusal asserted is `tier H2D batch partially refused: 1 of 34 items
(injected failure (MEMRA_KV_HOST_FAULT=contract-promote-reject))` and `1 of 32 items` respectively. On this tree
the promote cells refuse at the SETTLE for `postpublish` and `readyview` and at SUBMIT for `presubmit` and
`reject`, A's day-18 shape; the six cells' next promote reads `promote submitted off the tick`, `promote
published off the tick`, then the H2D receipt with `published retired acknowledged`.

**The whole-budget arm of the failure gate (task 2), run for the first time.** Day 21 asserted
`poolfull_refusal_whole_budget()` by pattern against lane D's banked day-8 log only. Today both door arms boot the
pool-full cell under `MEMRA_KV_HOST_TENANT_PCT=100` on the target card. The gate prints, verbatim:

    pool-full refusal arm: whole host budget (MEMRA_KV_HOST_TENANT_PCT=100 disarms the share cap; insert-path skip demote after the copy)
    ok: pool-full refusal is LOUD and named (skip demote: entry X MB > host budget B MB)
    KV-HOST-SPILL FAILURE GATE: ALL GREEN

and the server line matched (both arms, twice per boot): `[prefix-host] skip demote: entry 159.9MB > host budget
1MB`. 14 `ok` rather than 15: the `prefix_host_tenant_rejects >= 1` check belongs to the share-cap arm only (no
tenant reject exists when the cap is disarmed); the other 14 assertions are the same in both arms. The share-cap
arm (the default) read on the same tree, default-off: `pool-full refusal arm: tenant share cap 50% (server default
50; pre-copy evaporation, the image alone exceeds the share)` and the server line `[prefix-host] demote evaporated
at the tenant share cap before the D2H copy: 64 tokens, 159.9MB (50% of 1MB, MEMRA_KV_HOST_TENANT_PCT; model
gate); reclaim refused: the image alone exceeds the share (159.9MB > 1MB); nothing evicted`, 15 `ok`.

Observation from the ON whole-budget arm's server log (`failure-default-pct100-on/ev/poolfull-server.log`),
recorded, not acted on: under the door at `TENANT_PCT=100` the pool-full demote runs the whole Move 1 contract
before its refusal, `demote submitted off the tick: 64 tokens, 159.9MB, ticket seq=1, 34 items on the contracts
door's copy stream (model gate)`, `contracts door D2H receipt: ... items=34 ... complete=34 require=ok ... retired
acknowledged`, `demote published off the tick: ticket seq=1 complete after 1 poll(s), 184.4ms from submission to
completion (tick-top poll)`, and only then `skip demote: entry 159.9MB > host budget 1MB`: a 160 MB copy and a
full ticket lifecycle for an image that can never be resident in a 1 MB budget. This is the shape the code
chooses on purpose: `worker.rs` (the `host_bytes > host.budget` arm before the lease reservation) says an image
above the whole budget takes no ledger charge and is refused by name by `insert` after the copy "exactly as the
OFF arm does", so the OFF line stays the line (ruling 15). It is reachable only with the share cap disarmed, and
production runs the default 50, where the pre-copy evaporation refuses before any copy. Named for the lead as a
cost of the whole-budget arm under the door, not a defect of the gate or a change I made.

**The identity gate on Move 1 whole.** All four arms `ALL GREEN (teeth=0)`, 12 `ok` each. The ON arms' server
logs carry A's promote half: `promote submitted off the tick: 64 tokens, 158.9MB, ticket seq=2, 34 items on the
contracts door's copy stream; request parked`, `contracts door H2D receipt: ticket issuer=2 seq=2 epochs=0/1/1
items=34 (16 KV planes, draft) complete=34 require=ok checksums_sha256=5b58bfb9… published retired acknowledged`,
`promote published off the tick: ticket complete after 1 poll(s), 5.5ms from submission to completion (tick-top
poll)`; the plain ON arm the same with `items=32`. The OFF arms are the unchanged program. Teeth zero in every
arm: the promoted bytes equal the synchronous program's, which is A's identity law read on this lane's gate.

## Task 1, the local RTX 5090 Laptop GPU (the fault gate, both arms)

Tree `98170f182` for the first cell and `17756ec5b` for the second (the receipts commit landed between them; bash
and docs only, one server binary `d0a7d3f0e0c72051…` built from `98170f182` under `CPUQuota=1200%`,
`rtx5090-day22/build.log`, `Finished release in 3m 17s`), the Qwen3.5-9B NVFP4 MTP artifact,
`MEMRA_HOSTGATE_CACHE_MB=64`, each gate under its own `flock -n /tmp/memra-5090.lock`. The card was held by other
sessions when the battery started (another session's `memra-server` from `wt-525`, then a lane B server, seen only
by cwd in the driver's before snapshot, never signalled): the first cell waited 8 bounded retries of 120 s
(`fault-default/lock-retries.txt`) and ran 03:14:33Z to 03:15:28Z; the second followed at once. Receipts
`rtx5090-day22/<cell>/`, `battery.log`.

| Cell | Environment | Verdict line | ok / FAIL |
|---|---|---|---|
| fault-default | default (door ON by construction) | `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` | 65 / 0 |
| fault-plain | `MEMRA_SERVE_SPEC=0` (door ON by construction) | `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` | 65 / 0 |

Reject cell, verbatim: `promote-reject: the entry's plane count from the server's r2 D2H receipt: items=18`
(default) and `items=16` (plain), then `ok: promote-reject: the injected refusal's \`1 of M items\` M equals that
receipt's items=N` and `ok: promote-reject: the entry carries at least two planes, so the reject is partial
(items=N >= 2)` in both. The floor's margin on this class is 18 and 16 against 2.

Co-tenant note, stated: a `colbert-2/.venv/bin/python` process (1390 MiB, not a spill lane, not mine) appears in
the AFTER snapshot of `fault-default` and in both snapshots of `fault-plain`, so it sat on the card while my gate
held the lock; it was not touched. The cells are pass/fail gates with no timing claim, so the co-tenant changes
nothing they assert; it is recorded because the snapshots are part of the receipt. The whole-budget arm and the
failure and identity gates were not run locally today (the brief asked for the fault gate on the local card; the
27B receipts above stand for the whole-budget arm).

## Task 3: the HOSTPREFIX door review table

`HOSTPREFIX-DOOR.md`, "Review table for the decide-by": the fault gate row now names the floor and today's
receipts (`items=N >= 2`, 65 `ok`, both arms, both cards where run); the failure gate row now carries the
whole-budget arm as a run receipt (both door arms on the target card, 14 `ok`, the insert-path line verbatim)
instead of a pattern match against a banked log; the identity gate rows point at today's four arms on Move 1
whole. The "Still missing" list re-read:

1. The arena under the door: unchanged, scoped by ruling 28 (the arena lease handoff stays scoped until the
   decide-by review; one pinned budget or two is decided there with the door). Nothing today touches it.
2. The DFlash tail slice: the question survives day 20 and is not answered by it. Day 20 bounded the STANDALONE
   whole-prompt tap sink (`generate_spec_dspark` and `generate_spec_dflash` in `dflash.rs`: the prefill tap's
   buffer shape at prime time, memra#365). The tail slice is a different object: the host tier's image of the
   DFlash draft KV TAIL (`PrefixEntry.dspark_draft` into `HostPrefixEntry.dspark_draft`) under the contracts door,
   which is refused by name because no drafter artifact identity is derivable from a GGUF digest and no gate boots
   a DFlash drafter on the card. Day 20 changed neither fact. What day 20 did add is the input the identity would
   need: its artifact table records the export directory's byte manifest (`config.json` and `model.safetensors`
   sha256, equal to #370's `qualification.json` entries), so the tail slice's identity is derivable from the
   EXPORT DIRECTORY manifest, which is the binding the door does not have. Pre-registered, no cell exists.
3, 5, 6 unchanged (verify digest v3; the RTX 5090 class pair; the promote-side census question). Item 4 was
resolved day 21.

## Hygiene on the final tree

`cargo fmt --all -- --check` rc=0 on the merge tree (`day22-cpu/fmt-merge.log`; no Rust file changed after it),
`cargo test -p memra-server --lib` `786 passed; 0 failed; 14 ignored` (`day22-cpu/server-tests-merge.log`, the
settle-first and promote-order censuses included); `tools/check-flags.sh` no uncovered runtime names (no new
`MEMRA_*` read: the driver sets the existing `MEMRA_KV_HOST_TENANT_PCT`, which has its FLAGS.md row);
`tools/check-conflict-markers.sh` OK; `python3 tools/check-public-boundary.py check` 0 new (the receipts carry
`127.0.0.1` only, as day 21's did); `git diff --check` clean; shellcheck silent on the two drivers.

## Pushes

`cf45be796` (the two merges), `98170f182` (drivers), `17756ec5b` (target-card receipts), then the docs, each in
`MEMRA_RELEASE_QUALIFICATION_MODE=development` (printed `UNQUALIFIED DEVELOPMENT ... no GPU qualification
claimed`, logged in the clone's `.git/memra-gate-skips.log`). Not merged into main, no PR opened.

## Left as it was, and cleanup

Not touched: `/root/artifacts`, `/root/memra-spill`, other lanes' worktrees or processes (lane B's server held the
local card for part of the sitting, seen only in the driver's compute-apps snapshots; another session's
`memra-server` from `wt-525` held it before that). Cleaned: the shipped bundle on both ends, `/tmp/spill-c-day22`
(its logs banked under `day22-cpu/` and `rtx5090-day22/`), no server of mine running on either card, both locks
free at close; `/root/wt-c` left checked out at `98170f182`, clean. Open for the lead: the whole-budget arm's
copy-then-refuse cost under the door (above); whether `docs/TESTING.md` lines 1654-1655 should carry a pointer to
day 21 (unchanged from the day-21 note).
