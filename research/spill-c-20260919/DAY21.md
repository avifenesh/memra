# Session C day 21: the two red gate lines that were not the door's (the failure gate's pool-full cell, the fault gate's hardcoded item count), then the door's three gates on the tree that carries lane A's Move 1 slice

Lane `lane/spill-c-20260919`, checkout `wt-spill-c`. Every push today in the announced
`MEMRA_RELEASE_QUALIFICATION_MODE=development` mode (the #589 hook refuses the tree `UNQUALIFIED` because the
content-bound census sees the engine files lane A and main brought in; the skip is printed and logged; no GPU
qualification is claimed). Every cell below is `executed-not-qualified` development evidence. No commit on main,
no PR. No engine change today: the two fixes are in the two gate scripts.

## Merges (first action)

- `origin/main` `124dacc76` (#620, integ28: my day 20 on main) merged clean as `586a55e47`.
- `origin/lane/spill-a-20260919` `b967b8d30` (A day 17: Move 1's first slice under the door, the D2H demote on a
  second owner-thread stream with tick-top publication) merged as `b4cb89cec`. One conflict, `research/INDEX.md`:
  the HEAD side of the hunk was not content but the recursive merge's virtual-base marker block
  (`||||||| merged common ancestors`, `>>>>>>>>> Temporary merge branch 2`; neither parent carries a marker, checked
  on `origin/main`, A's tip and my day-20 tip), the other side A's `spill-a-20260919/day17` row. Resolution: the
  marker block dropped, A's row kept; `tools/check-conflict-markers.sh` OK before the commit. The lead integrates
  A separately; this lane carrying A's commits is so its gate runs read A's slice and my gate fixes together.

## Task 1: the failure gate's `pool-full refusal is LOUD and named` line. Verdict: the gate was stale; the server was right.

The cell (`tools/kv-host-spill-failure-gate.sh`, cell 1) boots with `MEMRA_KV_HOST_MB=1` so no real entry fits
the tier and asserts `grep -q "\[prefix-host\] skip demote: entry"` on the server log. That line is printed by
`HostPrefixCache::insert` (`crates/memra-server/src/worker.rs:8581`, `skip demote: entry {:.1}MB > host budget
{:.0}MB`), after the D2H copy. It is reached only when the per-tenant share cap is disarmed:
`tenant_cap_would_evaporate` (`worker.rs:8291`) returns `false` at `tenant_pct >= 100`, and otherwise the demote
path checks the cap BEFORE the copy (`worker.rs:11294`) and, when the image alone exceeds the share, prints the
typed line

    [prefix-host] demote evaporated at the tenant share cap before the D2H copy: 64 tokens, 54.8MB (50% of 1MB, MEMRA_KV_HOST_TENANT_PCT; model gate); reclaim refused: the image alone exceeds the share (54.8MB > 1MB); nothing evicted

and counts it in `prefix_host_tenant_rejects`. That line names the pool (the tenant's share of the host budget),
the requested bytes (54.8MB) and the budget (50% of 1MB, share 1MB). It is one line, typed, loud. Nothing on the
server side regressed.

Why the gate ever passed: `MEMRA_KV_HOST_TENANT_PCT` defaults to 50 (`worker.rs` `parse_kv_host_tenant_pct`,
`DEFAULT_KV_HOST_TENANT_PCT = 50`, "Default 50 BY DESIGN"), and both the default and the pre-copy check are in the
migration root `49d1d6f65` (2026-09-01); the FLAGS.md row calls the pre-copy evaporation line "the exact
production text" and the insert-path `skip demote` line "a defense-in-depth arm production demotes never reach".
The reclaim suffix (`reclaim refused: the image alone exceeds the share (X > Y); nothing evicted`) came with
`405466cf7` (2026-09-21, memra#384, lane A). Lane D's day 8 ran the gate green with `MEMRA_KV_HOST_TENANT_PCT=100`
set out of band (`research/spill-d-20260919/DAY8-CELLS.md:98`; its banked `poolfull-server.log` reads `tenant
share cap 100% = 1MB` and reaches `skip demote: entry 160.7MB > host budget 1MB`). Every default-environment run
of the gate since (my day 13 to 16, A's day 17, both cards, both door arms) read `1 FAILURE(S)` on this one line,
which I carried as "pre-existing" without reading the code under it. It was never the door's: the same line fails
with the door OFF.

Fix (gate side only, `7efab005d`, `tools/kv-host-spill-failure-gate.sh`): the gate mirrors the server's parse of
`MEMRA_KV_HOST_TENANT_PCT` (integer 1..=100, else 50) into `TENANT_PCT`, prints which refusal arm it asserts, and
asserts the FULL shape of the refusal the effective cap produces, anchored, with its bytes and its budget:
`poolfull_refusal_share_cap()` below 100 (`demote evaporated at the tenant share cap before the D2H copy: [0-9]+
tokens, [0-9]+\.[0-9]MB \([0-9]+% of [0-9]+MB, MEMRA_KV_HOST_TENANT_PCT; model gate\); reclaim refused: the image
alone exceeds the share \([0-9]+\.[0-9]MB > [0-9]+MB\); nothing evicted$`) plus a new check `the refusal counted
(prefix_host_tenant_rejects >= 1)`; `poolfull_refusal_whole_budget()` at 100 (`skip demote: entry
[0-9]+\.[0-9]MB > host budget [0-9]+MB$`). The match moved to follow the cited deliberate text (`49d1d6f65` default
and pre-copy check, `405466cf7` suffix), and it is stricter than the prefix it replaces, not looser. Self-test
before any run: A's banked `rtx5090-day17/failure-on/poolfull-server.log` matches the share-cap pattern on 2
lines with `prefix_host_tenant_rejects=2`; D's banked `day8/native/legacy-failures/state/poolfull-server.log`
matches the whole-budget pattern on 2 lines. Header rewritten to say all of this. The other 14 assertions are
unchanged.

## Task 2: the fault gate's `1 of 34 items`. Verdict: the gate was stale (a per-artifact constant); the server was right.

`tools/kv-host-contract-fault-gate.sh` `pcell promote-reject` carried the literal `tier H2D batch partially
refused: 1 of 34 items (injected failure (MEMRA_KV_HOST_FAULT=contract-promote-reject))`. The server prints
`{rejected} of {total} items{injected}` (`worker.rs:10477`) where `total = batch.items.len()`, the entry's plane
count: 34 on the 27B (16 KV planes x 2 plus the draft planes), 18 on the 9B. So on the local 5090 the cell's
literal never matched and five assertions of that cell failed (A's day 17: `5 FAILURE(S)`, all `promote-reject`).

Fix (`7efab005d`): `d2h_receipt_items()` reads `items=N` from the server's FIRST `contracts door D2H receipt`
line (r2's demote of E_A, the very entry r3 promotes into the injected refusal); `reject_total_matches_receipt()`
parses the injected refusal's `1 of M items` and asserts `M == N`; `pcell` takes the literal `partial-reject`
and, after the requests, builds the refusal from N. Two assertions added (`the r2 D2H receipt names the entry's
item total`, `the injected refusal's 1 of M items M equals that receipt's items=N`); every existing assertion
kept (12 per promote cell). No table in the gate. Self-test before any run: A's banked target-card
`promote-reject-server.log` reads `items=34` and refusal total `34`; A's banked local log `items=18` and `18`.

## Cells (both cards, one tree `9be3f7373` = `7efab005d` plus the drivers)

Driver `research/spill-c-20260919/day21-cell.sh <cell> <model> <bin> <root> [cache_mb] [lock_fd]`: one gate
under one named environment (`*-plain*` adds `MEMRA_SERVE_SPEC=0`, `*-on` adds `MEMRA_KV_HOST_CONTRACTS=1`;
the fault gate is door ON by construction so it has the two environments only), `nvidia-smi
--query-compute-apps` before and after, binary digest, tree SHA, verdict line, exit code, `status=
executed-not-qualified`. Locally the gate takes `/tmp/memra-5090.lock` itself (`flock -n`), a busy lock retried
15 x 120 s, the holder never signalled; `MEMRA_HOSTGATE_CACHE_MB=64` for the 9B (its 64-token entry is 53.8 MB;
the budget must hold one seed entry, not two). On the target card `day21-box-run.sh` waits for the build
receipt and runs each cell through `tools/tier-battery.py --rig pro-single --timeout 3600 --external-lock`
(`/tmp/memra-gpu.lock`, bounded retries 75 x 120 s), `MEMRA_HOSTGATE_CACHE_MB=256` for the 27B. Local binary
built from `b4cb89cec` under `CPUQuota=1200%` (`Finished release in 2m 49s`); target-card binary built from
the same tree (`pro-single-day21/build.log`, `rc=0`). The gate scripts are read at run time, so the fixed gates
run against the binary built from A's slice.

Both cards were held by other lanes when the batteries started (local: a lane B server on the card, seen by
its cwd in `--query-compute-apps`, not touched; target card: the collector's `REFUSED: [Errno 11] Resource
temporarily unavailable`); both drivers waited in their bounded retry loops.
