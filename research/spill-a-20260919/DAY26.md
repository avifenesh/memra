# WP-A day 26: proposal 1 implemented (the promoted-pin refusal of the restore route) and its acceptance gate on the target card

Lane `lane/spill-a-20260919`, checkout `wt-spill-a`. Every push today in the announced
`MEMRA_RELEASE_QUALIFICATION_MODE=development` mode (the #589 hook refuses the tree `UNQUALIFIED` because the
content-bound census sees engine files in the range; the hook prints `UNQUALIFIED DEVELOPMENT: ... no GPU
qualification claimed` and records the skip in the clone's `.git/memra-gate-skips.log`). Every cell below is
`executed-not-qualified` development evidence; nothing here is a qualification claim. No commit on main, no PR.

## Merge (first action)

`gh pr list --head lane/spill-integ41-20260922 --state all` printed `[]` and `origin/lane/spill-integ41-20260922`
does not exist on origin, so the first-action rule's `else` branch applied to the lead's branch where it exists: the
shared git dir carries `lane/spill-integ41-20260922` at `e14c270a6` (the lead's local worktree; the ref is a git
object, the worktree was not touched), which contains both `origin/main` `88d3dfd49` (#643, integ40) and my day-25
tip `483425d83`. Merged `--no-ff` as `71e46fabe`, clean (no conflict, `HOSTPREFIX-DOOR.md` came whole from the
integ41 side with the lead's rebuilt owed-cell table). The lead's integ41 record (ruling 36) is quoted from that ref.

## Pre-registration (this section is committed before any engine code moves)

**Ruling 36, as read from the integ41 record.** "Proposal 1 is approved for A day 26 as specified, with its acceptance
gate verbatim and one addition: the refusal's typed line names the pin and the entry so the gate can count it, and the
hit gate's ON arm on both cards must read the same route counts as day 24 minus exactly the promote-then-hit shape
(which the hit gate does not contain, so its counts must not move). Proposal 2 is accepted: the seam stays, 0.4 ms is
the recorded price, and Move 2 owed item 3 closes on this receipt."

**Proposal 1, exactly as DAY25.md states it.** In `host_restore_park_probe`, after `px.lookup` finds the hit `i` and
before the class check: if `hpx.promoted_pin` (the insertion pin of the promote published at THIS tick top, held until
the next tick top exactly so the parked request's re-admission finds its device hit, `host_promote_park_probe` and the
tick top's release) names entry `i`, return `false` with one typed line, and the request takes the OFF device-hit copy
on the tick. One predicate over existing state; no flag, no new state, no new engine seam, no numeric change (the OFF
program's copy, byte-identical destination).

**Ruling 36's addition, applied.** The typed line names the pin and the entry so a gate can count it:
`[prefix-cache] restore not routed (contracts door): the entry was promoted for this admission (insertion pin id=P,
N tokens, model M[, namespace "ns"]); the tick program copies it`. The pin's id is the device entry's id
(`PrefixPin { key, id }`); the entry's tokens and pool key (model, namespace) come from the entry `i` the lookup found.
The promote's ticket seq is NOT at hand in the probe: `PrefixPin` carries no ticket and the promote's contract is
retired at publication; carrying the seq onto the pin would be new state, which the proposal excludes, so the line
does not print it. The wording is chosen so that no existing counter moves on it: the hit gate counts
`refused (contracts door)` and `restore refused` (`tools/spec-on-cache-hit-gate.sh` lines 794 and 795) and neither
substring occurs in it; `stall_cell.py` keeps it in `server_log_lines` (the filter takes every `[prefix-cache]
restore` line), so the reader counts it per run.

**The one-program law.** The refused request takes the tick program's device-hit copy (`prefix_restore` in `admit`),
which is the OFF arm's program and the program every existing typed refusal of the probe already routes to; the
identity gate's ON arm covers it (a request never crosses programs mid-flight: the refusal happens in the admission
pass before any token). Fail-closed arms are untouched: the latch, the tenancy check and the class check keep their
order after the new predicate (the predicate sits before the class check, as the proposal states; a promoted entry
that the class check would refuse anyway is refused by the promoted-pin line first, which is the same outcome).

**Where the shape occurs in tonight's gates (read from day 24's logs before the run).** The hit gate has no promote
in either boot (`qwen-on-server.log` and `qwen-off-server.log` of `pro-single-day24/box/gates/hitgate-on/`: zero
`[prefix-host] promote` lines), so per ruling 36 its counts must not move: spec-on `capture_submitted=12
capture_published=12 restore_submitted=13 restore_landed=13 refused_contracts_door=0 restore_refused=0 latched=0`,
spec-off `capture_submitted=2 capture_published=2 restore_submitted=3 restore_landed=3`, `30 route submission(s)`,
11 spec-boundary captures with the draft plane, and zero `restore not routed` lines. The identity gate's ON arm DOES
carry the shape on this tree (day 24's `identity-default-on/host-on-server.log`: `promote published` 1, then `restore
submitted` for the same request at its re-admission, `request parked` 3, a second routed hit later): its route lines
are expected to read `promote published` 1, `restore submitted` 1, `restore landed` 1, `request parked` 2 and one
typed line; the gate asserts nothing on those lines (its clauses are the byte identity, the receipts' `items`, the
demote and promote lines), so `ALL GREEN (teeth=0)` is the gate and the line counts are a reading. (C's day-25
identity run on the integ36 tree `913095404` read `restore submitted` 0 after its promote because that tree did not
yet carry A's day-23 draft-bearing route; on the spec path the class check refused the draft entry silently. Not
this tree.) The fault gate's four promote cells carry the shape too (day 24: `restore submitted` 1 after the second
promote); the fault gate counts `restore submitted` only in its `d2d-restore` cell, which has no promote, so those
cells' lines are a reading and `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` is the gate. The twin gate has neither a
promote nor a routed restore.

**Acceptance gate, verbatim from DAY25.md.** "This cell again (promote arm, ON against OFF, twenty interleaved boots,
N=5 per arm per order, both orders, one hold) with the clauses (1) `request parked` per ON promote run reads 1 and
`restore submitted` reads 0, 100 of 100; (2) the request's ON e2e median drops by the re-admission median within
the pair's unc (221.4 to about 131; `on_minus_off` from +105.9 to about +15, the first park's own share); (3) the
tenant's ON stall stays within IQR of 149.4 (the second park cost the tenant nothing, so removing it must move
nothing; a move either way is a finding); (4) the day-21 restore arm (`--mode restore`, a hit on an entry NOT
promoted this admission) still parks once and lands 100 of 100 with `restore landed` lines; (5) the hit gate OFF and
ON `ALL GREEN` with every `spec==plain byte identity`, the identity gates OFF and ON `ALL GREEN`." STATE.md's
shorter form: "`request parked` 2 to 1 and `restore submitted` 0 in 100 of 100 ON promote runs; e2e down by the
re-admission median, about 221 to 131; the tenant's stall within IQR of 149.4, a move either way is a finding; the
day-21 restore arm still parking once and landing 100 of 100; hit and identity gates ALL GREEN in both arms."

**How each clause is read (`day26-reading.py`, fixed here; nothing tuned after the run).** (1) per ON promote run of
the double-park receipts: `request parked` == 1, `restore submitted off the tick` == 0 and `restore not routed
(contracts door)` == 1, in 100 of 100. (2) per order, `on_minus_off` of the request's e2e medians (50 runs per arm
per order) against the expected `+105.85 - 90.1 = +15.75` (day 25's delta minus its re-admission median), PASS when
`|on_minus_off - 15.75| <= unc` with unc the pair's quadrature of the two IQRs. (3) per order, the ON stall cell
median against 149.4 with day 25's ON IQR 0.1: `within IQR` or `FINDING`, printed with the decomposition (the tenant's
two largest gaps, the demote's `in - completion`, the tick); never a FAIL. (4) one ON boot of the day-21 restore cell
(`MEMRA_PREFIX_CACHE_MB=1024 MEMRA_KV_HOST_MB=8192 MEMRA_KV_HOST_CONTRACTS=1`, `--mode restore --n 50`: 100 timed
restore runs), per run `request parked` == 1, `restore submitted` == 1, `restore landed` == 1, `restore not routed`
== 0, replay PASS, `errors=0`. (5) both hit gate arms `ALL GREEN`, and the ON arm's census lines, route submission
count, spec-boundary draft-plane capture count and `restore not routed` count (0) equal to day 24's as listed above.
The identity gates' verdict lines are read from `gates/identity-*.log`; the identity ON route lines and the fault
gate's promote cells' lines are printed as readings.

**What the code says clause (3) will do (read before the run, to be confirmed by line).** Day 25's tenant stall is
`max ITL - p50`; its largest gap (162.8 median) is tick B, which carries the demote's two hashes (74.9), the restore's
poll and re-admission, then the re-admitted hit's suffix prime and the decode; the second gap (22.6) is tick A (the
decode, the promote's publication 6.5, the restore's submit). Under the refusal the hit is admitted in tick A's
admission pass and its suffix prime runs on tick A, so tick A grows by the device-hit copy (about 0.16 ms) plus the
suffix prime, and tick B shrinks by the prime and the restore's poll, install and take. If the prime is of the order
OFF's largest gap implies (OFF 98.7 = decode 13.4 + on-tick promote 10.8 + on-tick demote 6.2 + copy + prime, so the
prime is about 68 ms), the two stretched ticks come out near 88 and 90 and the tenant's stall would read about 76,
far below 149.4: a FINDING under clause (3) as DAY25 wrote it ("a move either way is a finding"), and the reading
that the second park DID cost the tenant, not through its own on-tick share (inside the +1.8 residual, as day 25
read) but by placing the hit's prime on the hash tick. If instead the stall stays within 0.1 of 149.4, the prime is
not where this reading puts it and the decomposition says where. Either outcome is reported with the numbers;
nothing is tuned on seeing them.

**The sitting (`pro-single-day26/driver.sh` on BOX3, one RTX PRO 6000 Blackwell, 600 W, the collector's lock).** In
order, one sitting: `double-park.sh` (the day-25 script byte-for-byte in shape, one hold, twenty boots, the same
harness `stall_cell.py`), `restore-arm.sh` (one hold, one ON boot, 100 timed restore runs), `gates.sh` (the day-24
set: identity default and plain OFF/ON, failure OFF/ON, the fault gate all cells, twin OFF/ON), `hitgate.sh` (its own
`flock`, OFF then ON), `unit-cells.sh` (option_b, option_c, the engine's `d2d_` cells). Bounded lock retries (60 x
120 s); lane C runs on the local 5090 today, so the card is expected mine; any holder is never inspected or
signalled. Local RTX 5090: the lead and lane C use it today; the hit gate ON arm there (ruling 36's "both cards") is
attempted after the box sitting behind `flock /tmp/memra-5090.lock` with a bounded wait, else recorded NOT RUN.

**Budget plan.** Reading, the merge and this pre-registration 0.9 agent-hours; the predicate, the typed line, the
census and table tests, the CPU checks 0.6; the box shipping, build and the sitting (twenty boots near 20 minutes,
the restore arm near 5, the gates near 25, the hit gate near 2, the unit cells near 3) 1.2 polled; the receipts, the
reading and the records 0.9. Total planned 3.6 of 4.
