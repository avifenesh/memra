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

## The code (`e008bf502`; the test-only clippy fix `55c56e8ee` changes no release source)

`host_restore_promoted_this_admission(px, hpx, pool_key, i) -> Option<u64>`: `hpx.promoted_pin` names entry `i` of
this pool when the pin's key equals the pool key and `px.id_index(pin) == Some(i)`; returns the pin's id (the device
entry's id). In `host_restore_park_probe` it sits between `px.lookup` and the class check; on `Some` the probe prints
the one typed line (`[prefix-cache] restore not routed (contracts door): the entry was promoted for this admission
(insertion pin id=P, N tokens, model M); the tick program copies it`) and returns `false`, so the request takes the
tick program's device-hit copy in the same admission. No flag, no new state, no numeric change; `unsafe` none. CPU:
`restore_probe_promoted_pin_names_only_the_promoted_entry` (the table: no pin; the pin names entry 0; the pin names
entry 1; the pin names another pool's entry 0; a pinned entry removed from the index is never named) and
`day26_promoted_pin_refusal_is_named_before_the_class_check` (lookup, then the refusal, then the class check, then the
route's own pin; the typed line's head and tail on one line; `return false` inside the refusal; neither hit-gate
counter substring in the printed text; one call site, one definition, no `MEMRA_` read in either). Battery on this
tree: `cargo fmt --all -- --check` clean; clippy `-D warnings` all targets on tier, engine and server `Finished`; the
`DOCS_RS=1 --target x86_64-unknown-linux-gnu` pass `Finished`; server lib `811 passed; 0 failed; 14 ignored`;
`check-flags: no uncovered runtime names`; markers `OK`; `git diff --check` clean.

## The sitting (target card, BOX3, one RTX PRO 6000 Blackwell, 600 W; `pro-single-day26/box/`)

Tree `e008bf502` on `/root/wt-a` (branch `lane-a-day26`, clean), binary `ca81aa7a1322f868…` (`ev/binary.sha256`),
harness `stall_cell.py` SHA-256 `13867e77da40a9b5…` (day 25's byte-for-byte). Four collector holds in one sitting,
12:55Z to 13:36Z, zero lock retries (`lock-retries.log` empty; the card was mine alone: `compute-apps.before.csv`
header only, `compute-apps.after.csv` header only), every `CELL.jsonl` `status: executed-not-qualified`,
`qualification: false`, `exit_code 0`: `double-park` 1171.7 s (twenty boots 12:55:49Z to 13:15:20Z; 32 to 52 C,
31.98 to 360.81 W, 0 to 17,109 MiB, 4,672 samples at 250 ms), `restore-arm` 483.7 s (41 to 54 C, 87 to 494 W),
`gates` 485.6 s (38 to 62 C, 86 to 510 W, up to 21,939 MiB), `unit-cell` 164.1 s; the hit gate under its own
`flock` 13:32Z to 13:33Z. Every double-park receipt `STALL REPLAY: PASS` (20 of 20), `errors=0`, one tenant text SHA
per boot, `server_promote_ms` count 10, no bad line: `DAY26 DOUBLE-PARK ADMISSIBLE all_receipts=True`.

## The reading (`day26-reading.py`, verbatim)

```
DAY26 CLAUSE 1 arm=on N_runs=100 parked_per_run=[1] restore_submitted_per_run=[0] not_routed_per_run=[1] runs_with_parked_1_submitted_0_not_routed_1=100 -> PASS
DAY26 CLAUSE 1 context arm=off N_runs=100 parked_per_run=[0] not_routed_per_run=[0]
DAY26 TYPED LINE (one ON run, verbatim): [prefix-cache] restore not routed (contracts door): the entry was promoted for this admission (insertion pin id=2, 64 tokens, model gate); the tick program copies it
DAY26 CLAUSE 2 e2e order=o1 on_median=206.8 off_median=115.3 on_minus_off=+91.4 unc=1.2 expected=+15.8 (day 25: +105.85 minus 90.1) |d-expected|=75.7 -> FAIL
DAY26 CLAUSE 3 stall order=o1 on_stall_medians=[81.9, 81.8, 81.8, 81.8, 81.8] on_cell_median=81.8 IQR=0.0 off_cell_median=85.3 on_minus_off=-3.4 unc=0.1 -> isolated | against day 25's 149.4 (IQR 0.1): -67.6 -> FINDING (moved, reported with the decomposition, not tuned)
DAY26 CLAUSE 2 e2e order=o2 on_median=206.7 off_median=115.4 on_minus_off=+91.4 unc=1.1 expected=+15.8 (day 25: +105.85 minus 90.1) |d-expected|=75.6 -> FAIL
DAY26 CLAUSE 3 stall order=o2 on_stall_medians=[81.8, 81.8, 81.9, 81.8, 81.8] on_cell_median=81.8 IQR=0.0 off_cell_median=85.3 on_minus_off=-3.4 unc=0.1 -> isolated | against day 25's 149.4 (IQR 0.1): -67.6 -> FINDING (moved, reported with the decomposition, not tuned)
DAY26 DECOMPOSITION arm=on N_runs=100 parked_per_run=[1] restore_readmission=[] promote_completion median=19.6 promote_in median=26.1 demote_completion median=97.5 demote_in median=172.3 demote_in-completion median=74.8 idle_p50(tick)=13.47 | tenant top gaps: largest median=95.3 second median=92.4 sum median=187.8
DAY26 DECOMPOSITION arm=off N_runs=100 parked_per_run=[0] promote_in median=10.7 demote_in median=6.2 | tenant top gaps: largest median=98.6 second median=16.6
DAY26 CLAUSE 4 restore-arm arm=on replay=PASS N_runs=100 errors=0 parked_per_run=[1] submitted_per_run=[1] landed_per_run=[1] not_routed_per_run=[0] runs_parked_1_submitted_1_landed_1=100 cached_tokens=[5088] stall_median=80.4 -> PASS
DAY26 CLAUSE 5 hitgate-off: SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen) (ok=61 FAIL=0)
DAY26 CLAUSE 5 hitgate-on: SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen) (ok=68 FAIL=0)
DAY26 CLAUSE 5 hitgate-on census: door lines: armed=1 door_on=1 capture_submitted=12 capture_published=12 restore_submitted=13 restore_landed=13 demote_submitted=0 promote_submitted=0 refused_contracts_door=0 restore_refused=0 latched=0
DAY26 CLAUSE 5 hitgate-on census: door lines: armed=1 door_on=1 capture_submitted=2 capture_published=2 restore_submitted=3 restore_landed=3 demote_submitted=0 promote_submitted=0 refused_contracts_door=0 restore_refused=0 latched=0
DAY26 CLAUSE 5 hitgate-on counts against day 24: spec_on_census_equal=True spec_off_census_equal=True route_submissions=30 (day 24: 30) spec_boundary_captures_with_draft_plane=11 (day 24: 11) not_routed_lines=0 (must be 0: the gate has no promote-then-hit shape) -> PASS
DAY26 READING identity-default-on route lines: promote_published=1 restore_submitted=1 restore_landed=1 parked=2 not_routed=1 hits=2 (day 24's tree: 1, 2, 2, 3, 0, 2)
DAY26 READING identity-plain-on route lines: promote_published=1 restore_submitted=1 restore_landed=1 parked=2 not_routed=1 hits=2 (day 24's tree: 1, 2, 2, 3, 0, 2)
DAY26 READING fault promote-presubmit: promote_published=1 restore_submitted=0 parked=1 not_routed=1
DAY26 READING fault promote-postpublish: promote_published=1 restore_submitted=0 parked=2 not_routed=1
DAY26 READING fault promote-readyview: promote_published=1 restore_submitted=0 parked=2 not_routed=1
DAY26 READING fault promote-reject: promote_published=1 restore_submitted=0 parked=1 not_routed=1
DAY26 GATE identity-default-off: KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0) (ok=12)
DAY26 GATE identity-default-on: KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0) (ok=12)
DAY26 GATE identity-plain-off: KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0) (ok=12)
DAY26 GATE identity-plain-on: KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0) (ok=12)
DAY26 GATE failure-off: KV-HOST-SPILL FAILURE GATE: ALL GREEN (ok=15)
DAY26 GATE failure-on: KV-HOST-SPILL FAILURE GATE: ALL GREEN (ok=15)
DAY26 GATE contract-fault: KV-HOST-CONTRACT-FAULT GATE: ALL GREEN (ok=93)
DAY26 GATE twin-off: PREFIX-NEWEST-TURN-FITS: ... V1=ok V2=ok V3=ok V4=ok V5=ok V6=ok -> PASS
DAY26 GATE twin-on: PREFIX-NEWEST-TURN-FITS: ... V1=ok V2=ok V3=ok V4=ok V5=ok V6=ok -> PASS
DAY26 ACCEPTANCE clauses (1) (2) (4) (5) = ['PASS', 'FAIL', 'PASS', 'PASS']; clause (3) is a finding either way (see CLAUSE 3 lines)
```

Unit cells (`unit/`): server `option_b_`/`option_c_` `test result: ok. 8 passed; 0 failed`; engine `d2d_` `test result:
ok. 5 passed; 0 failed`, cell (v)'s line `D2D-RECEIPT PRICE order=copy-first bytes=165675008 n_per_arm=5 ...
copy_median=0.156 ...` printed again on this card (its figure is day 22's row, not re-read here).

## Verdicts, clause by clause

**(1) PASS.** 100 of 100 ON promote runs: `request parked` 1, `restore submitted` 0, the typed line 1. The promote
arm's hit no longer parks a second time; the OFF arm is unchanged (`parked_per_run=[0]`, no typed line, the code
path is behind the door's `transfers` check).

**(2) FAIL, and the reading of why.** The request's ON e2e median fell from 221.4 to 206.8 (order 1) / 206.7 (order
2): 14.6 ms, not the re-admission median 90.1 the clause expected; `on_minus_off` +91.4 against the expected +15.8,
`|d - expected| = 75.7`, far outside unc 1.2. The 75.7 is the demote's two on-tick hashes (`demote_in -
demote_completion` 74.8 median, 100 of 100). Where they sit in the request's path, read from the code after the
result: `advance_sample_emit` (worker.rs, "the decode tick's HOST half: sample from last_logits, emit the token, run
the stop battery") samples a session's token from the logits the PREVIOUS step produced, so a `max_tokens=1` request
whose prime ran in tick A's step receives its token, its `MaxNew` stop and its response in tick B's host half, after
tick B's top has polled the demote and its publication has run the two hashes. On day 25 the prime ran in tick B's
step (after the hashes) and the token came in tick C: the hashes were inside the 90.1 ms re-admission wait, paid once.
Under the refusal the prime runs in tick A and the hashes still sit between the prime and the token: paid once again.
What the refusal removed from the request's path is tick A's decode (13.47) plus the poll-to-re-admit slack (about
1.8): 15.3 by arithmetic, 14.6 observed (the pair's unc 1.2). By arithmetic the ON request's +91.4 over OFF is the
first park's own share 15.3 (the off-tick promote's poll 19.6 plus publish 6.5 against OFF's on-tick 10.7) plus the
hashes 74.8 plus 1.3. The clause's premise, written on day 25, was that the re-admission wait would leave the
request's path whole; its hash share does not, because the token is emitted a tick after the prime. The clause is
reported FAIL as pre-registered; nothing is re-derived to pass it. Any change that would remove the remaining 74.8
from the request's path is a scheduler program (where the token of a prime-finished request is emitted, or where the
demote's hashes run relative to the host half), outside Move 2's door and outside this lane's proposals; named, not
proposed.

**(3) FINDING: the tenant's stall fell from 149.4 to 81.8 (IQR 0.0, both orders), 67.6 ms, and now reads 3.4 ms
BELOW the OFF arm's 85.3 (`on_minus_off=-3.4 unc=0.1 -> isolated`, both orders).** Day 25's clause said "the second
park cost the tenant nothing, so removing it must move nothing"; the code reading in this file's pre-registration
said the opposite and the numbers agree with the reading. The decomposition: the tenant's two stretched ticks read
95.3 (largest, median) and 92.4 (second), sum 187.8; day 25's were 162.8 and 22.6, sum 185.4; OFF's 98.6 and 16.6,
sum 115.2. The total stretched work per promote-then-hit is unchanged within 2.4 ms (187.8 against 185.4), but it is
now split: tick A carries the decode 13.47, the promote's publication 6.5, the device-hit copy (about 0.16) and the
re-admitted hit's suffix prime (about 72), 92.4; tick B carries the decode, the demote's two hashes 74.8 and the
finished request's token, stop and retire (about 7), 95.3. The stall is `max ITL - p50`, so it reads the larger tick:
95.3 - 13.47 = 81.8. On day 25 the second park placed the prime on the hash tick (74.9 + 72 on one tick, 162.8) and
that stacking, not the park's own on-tick share, was the tenant's 149.4. The ON stall now sits below OFF because OFF
stacks its on-tick promote 10.7, demote 6.2 and the prime on ONE tick (98.6) while ON's largest tick is 95.3; the sum
of ON's two gaps against OFF's (187.8 against 115.2) still says the door costs the tenant 72.6 ms of tick time per
promote-then-hit, the two hashes, spread over two ticks instead of stacked. Two consequences for the ledgers: day 25's
"the tenant's +64 is the demote's hashes on one tick, not the second park" stands for the +64 against OFF, but its
"the second park cost the tenant nothing" is refuted (the park decided which tick the prime landed on); and the
demote's `in` figure grew from 97.2 to 172.3 because its completion poll now waits tick A's longer step
(`demote_completion` 22.3 to 97.5) while its on-tick work is unchanged (`in - completion` 74.9 to 74.8): the `in`
figure is submission-to-publication, not on-tick cost, and the ledgers that quote it as a cost must say so.

**(4) PASS.** The day-21 restore arm, one ON boot, 100 timed restore runs, `replay=PASS`, `errors=0`: every run
`request parked` 1, `restore submitted` 1, `restore landed` 1, `restore not routed` 0, `cached_tokens=[5088]`; a hit
on an entry NOT promoted this admission still takes the route. Stall median 80.4 on this shape (context only, not a
pair).

**(5) PASS.** Hit gate OFF `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (61 ok), ON `ALL GREEN (qwen)` (68 ok, 0
FAIL), and per ruling 36 the ON arm's counts equal day 24's exactly: spec-on `capture_submitted=12
capture_published=12 restore_submitted=13 restore_landed=13 refused_contracts_door=0 restore_refused=0 latched=0`,
spec-off `2 / 2 / 3 / 3`, `30 route submission(s)`, 11 spec-boundary captures with the draft plane, zero typed
refusals (the gate has no promote-then-hit shape). Identity default and plain OFF and ON `ALL GREEN (teeth=0)` (12
ok each); the identity ON arms' route lines read exactly the pre-registered `1, 1, 1, 2, 1, 2`. Failure OFF and ON
`ALL GREEN` (15 ok); contract fault `ALL GREEN` (93 ok); twin OFF and ON `-> PASS`; the fault gate's four promote
cells read `restore_submitted=0 not_routed=1` (a reading; the gate's `d2d-restore` cell, which counts `restore
submitted`, has no promote and stayed at 1).

**The gate as a whole.** Clauses 1, 4 and 5 pass; clause 2 fails on the premise stated above; clause 3 is a finding
(a 67.6 ms fall, below OFF). The mechanism does what proposal 1 said (one park, no restore submission, the tick copy,
no counter moved where the shape is absent); the request-latency saving it delivers is 14.6 ms, not 90; the tenant's
stall saving it delivers, 67.6 ms, was not what day 25 predicted. Nothing was tuned or relaxed on seeing the numbers.
The code stays on the branch as landed; whether the refusal is kept, with the gate's clause 2 re-derived from the
token-emission reading, is the lead's ruling, not this lane's.

## Local RTX 5090 (ruling 36's "both cards")

`rtx5090-day26/hitgate-5090.sh`: the hit gate OFF then ON with the day-26 tree's local release binary and the 9B
artifact under the gate's own `flock /tmp/memra-5090.lock`, after a bounded idle wait (15 x 120 s per arm; the lead
and lane C use the card today; the holder is read from `nvidia-smi` only, never signalled). Outcome: see
`rtx5090-day26/battery.log` and the sentence below (filled when the arm lands or is recorded NOT RUN).

5090 OUTCOME: pending at the time of this section's first commit.
