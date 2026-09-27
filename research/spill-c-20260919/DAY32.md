# Session C day 32: the `MEMRA_ADMIT_BY_MEMORY` decision packet, the `MEMRA_KV_PARK_COMPACT` row reconciliation, door table pointers

Lane `lane/spill-c-20260919`, checkout `wt-spill-c`. Every push today in the announced
`MEMRA_RELEASE_QUALIFICATION_MODE=development` mode (the hook prints `UNQUALIFIED DEVELOPMENT: refs/heads/lane/spill-c-20260919
at <sha>; no GPU qualification claimed` and records the skip in the clone's `.git/memra-gate-skips.log`). No card work
today: no server booted, no lock taken, no GPU process of mine. No engine change: records and one CPU test run. No
commit on main, no PR. Every number below is read from a receipt file or is labelled arithmetic; nothing here is a
qualification claim, and nothing here recommends an outcome.

## Merge (first action)

`git fetch origin && git merge --no-ff origin/main` (main `f10973ab7`, #644: this lane's day 30, A day 25, the lead's
integ41 record) merged CLEAN as `04c67b2e0` (no conflict; the door table and the lead record are the integ41 side by
construction). `origin/lane/spill-integ42-20260922` does not exist (`git branch -r | grep integ42`: nothing), so the
second merge did not apply. Pushed in the announced development mode (`ffe9534a0..04c67b2e0`).

## 1. The `MEMRA_ADMIT_BY_MEMORY` decision packet (`ADMIT-BY-MEMORY-DECISION-PACKET.md`, `8a3ea6b79`)

Written in the shape of `DOOR-DECISION-PACKET.md`: the question, what the door is today from the code, B's evidence
per card re-read from the receipt files, the gates covering the ON arm verbatim per card, what is unmeasured or
unbuilt, the three hygiene outcomes with what each requires, and an appendix with the tracing rule, the numbers
deliberately not carried, the untraced list (none) and the missing list (six). Findings of the assembly, stated:

1. **Part (b) is the tick program under both doors.** `evict_all_demoting` (`worker.rs:11978-12027`) calls
   `host_demote_prefix_ref(engine, host, entry, ContractD2h::OnTick)`, so with `MEMRA_KV_HOST_CONTRACTS=1` the
   admission flush's demote is the day-16 synchronous program (Move 1 owed item 3 of this lane's table), and with the
   host tier unarmed (`MEMRA_KV_HOST_MB` unset) `host.armed()` is false and (b) is `PrefixCache::evict_all` exactly. The
   door packet's item 6 said "the door's route when ON"; the packet now says the route by name.
2. **Every measurement B banked is of the OFF arm.** All six day-26 cells and both day-27 after cells print
   `[admit-mem] door=OFF open_output_tokens=8192 defer_budget_ms=8000 (...)` (one per `server.log`, counted); the runner
   never sets the door. The 129x is `arm=i L0 ... median=129.26` on the target card in four receipts
   (`B/pro-single-day26/cells/{ab,ba,warm}/REPORT.txt`, `B/pro-single-day27/cells/after-ab/REPORT.txt`), identical to
   the digit because the prompts and the greedy program are the same. Regimes recomputed from the CSVs today:
   `ab` 778 samples 32 to 64 C, 32.40 to 512.22 W; `ba` 777 samples 44 to 64 C, 52.07 to 511.50 W; `warm` 731 samples
   37 to 65 C, 33.43 to 512.74 W. Local (9B, `MEMRA_CTX=65536`): `arm=i L0 ... median=11.30`, one binary across the
   three cells (`binary.sha256` equal; the `source.txt` stamps differ because B committed records between cells).
3. **No `[admit-mem] id=... verdict=...` decision line exists in any tracked receipt** in memra or darklanes. The one
   boot with `door=ON` that has a receipt (darklanes `research/glm5b200-stress-criterion-20260911/receipts/slot-A.log`,
   a 2x B200 TP-2 boot whose subject was another criterion) printed the boot line and never reached a decision: 0
   decision lines, 0 `reclaim demoted`, 0 status-429 rows in that directory.
4. **The FLAGS row's receipt pointer does not resolve.** `research/glm5-1m-b200-ship-20260906/` in darklanes holds no
   file naming the door or `[admit-mem]` (grep 0); the door's lane file is `research/glm5-memory-admission-20260909/LANE.md`,
   whose requalification cell section is titled "(not yet run)". The B200 bring-up card cell's only record is the
   quoted line `[cell] 1x900k + 8x4k resident: free 177.70 -> 164.39 GiB (13.31 GiB charged)` in that file and in the
   `ad02b8c9a` commit message; no receipt file exists in either repo.
5. **The door landed 2026-09-10 (`ad02b8c9a`, #431, v0.138.0); the decide-by is 13 days after landing.**

CPU gate on today's tree (`day32/admit-cpu-tests.log`, `day32/CELL.txt`; tree `04c67b2e0`; `systemd-run --user --scope
-p CPUQuota=1200% -p MemoryMax=28G cargo test -p memra-server -- admit_memory memory_admission_door the_open_output_arm`):
`test result: ok. 13 passed; 0 failed; 0 ignored; 0 measured; 810 filtered out; finished in 0.01s` (the 11
`admit_memory::tests` and the 2 `worker::tests` wiring tests, each `... ok`). The `#[ignore]` card cell was not run (a
B200-geometry cell needing >= 32 GiB free under an exclusive lock; not a one-boot 5090 cell). No other number was
produced today; the six numbers the decision would need and has no receipt for are listed in the packet's appendix D,
not run, per today's rule (B's pre-registered cell is a two-card, two-order, N>=5 hold, not one boot).

## 2. The `MEMRA_KV_PARK_COMPACT` row: which text is authoritative, and the proposed reconciliation (for the lead)

**What the two texts say.** `docs/FLAGS.md:324` (after B's day-28 edit `72f89e233`, on `main` since #643): the VALUE
column reads `**0 = OFF by design**` and the prose, near its end, reads `**decide-by: 2026-10-06** (set 2026-09-22 by
WP-B day 28, 14 days after the door's first serving receipt; the date only, the verdict is the owner's)`.
`research/spill-b-20260919/KV-RESIDENCY-DESIGN.md` "Day 28 addendum": "`MEMRA_KV_PARK_COMPACT` (default OFF) had no
`decide-by:` date; the door-hygiene rule wants one on every default-OFF door. Set 2026-09-22 in `docs/FLAGS.md`:
**decide-by 2026-10-06**, 14 days after the door's first serving receipt (`DAY27.md` 2.6). The date is the lane's; the
verdict is the owner's." So the two documents AGREE on the date; the conflict is inside the FLAGS row itself, between
the value column and the prose. (This lane's packet item 6 read the row as it was before `72f89e233`; the row on today's
`main` already carries the date in its prose.)

**Which is authoritative.** `docs/FLAGS.md` is the registry the hygiene rule names ("Every default-OFF door lands with a
`decide-by:` date in its `docs/FLAGS.md` row"); a lane record is evidence, not the register. Inside the row, the dated
sentence is the operative one: the rule exempts a default-OFF switch from a date only when "its row says so" (the
catalog's explicit forms are `MEMRA_AUDIO_DRIVE`'s "RED-ARM DOOR, kept deliberately, no decide-by" and
`MEMRA_SPEECH_THREADS`'s "no decide-by: this is a ... knob, not a serving door"), and this row gives no exemption
reason; its "OFF by design" describes where the default came from (the kv-tenancy lane, "unmeasured on serving
hardware"), not an exemption. The row's own "Default OFF because the mechanism is unmeasured on serving hardware ...
PENDING" sentence is also stale by one clause since day 27 (the park-time copy cost has a target-card serving receipt,
which the same row quotes further down). No tool parses the value column or the `decide-by:` token
(`grep -rn decide-by tools/ .github/workflows/`: nothing), so the reconciliation is prose-only and moves no gate.

**Proposed row text (exact; the lead edits `docs/FLAGS.md`, this lane does not).** Three replacements inside line 324,
everything else in the row unchanged:

1. Value column: replace `| **0 = OFF by design** |` with `| **0 (default OFF), decide-by: 2026-10-06.** |` (the form
   of the `MEMRA_ADMIT_BY_MEMORY` and `MEMRA_PRIME_ATTN_FA2` rows).
2. In the prose, replace `**decide-by: 2026-10-06** (set 2026-09-22 by WP-B day 28, 14 days after the door's first
   serving receipt; the date only, the verdict is the owner's).` with `The decide-by was set 2026-09-22 by WP-B day 28,
   14 days after the door's first serving receipt (`research/spill-b-20260919/DAY27.md` 2.6); the date is the lane's,
   the verdict the owner's.`
3. Optional, the stale clause: replace `Default OFF because the mechanism is unmeasured on serving hardware: the resume
   byte-identity gate (compacted-park vs plain-park, both resume shapes), the step-OOM adjacency replay (the multi-active
   OOM-park incident is adjacent code), and the park-time copy-cost receipt are pod-battery cells, PENDING
   (`research/kv-tenancy-20260831/REPORT.md`).` with `Default OFF: the resume byte-identity gate (compacted-park vs
   plain-park, both resume shapes) and the step-OOM adjacency replay (the multi-active OOM-park incident is adjacent
   code) are still PENDING (`research/kv-tenancy-20260831/REPORT.md`); the park-time copy cost has a target-card serving
   receipt (WP-B day 27, below) and the local 9B pair is owed.`

If the lead rules instead that the door IS an owner-designed switch exempt from a date, the reconciliation runs the
other way: the value column keeps `0 = OFF by design`, the prose says why it is exempt in the catalog's explicit form,
and the `decide-by: 2026-10-06` sentence and the design doc's addendum are withdrawn together. This lane proposes the
text of both directions and chooses neither.

## 3. Door table pointers and records

- `HOSTPREFIX-DOOR.md` section E: a DAY 32 paragraph pointing at `ADMIT-BY-MEMORY-DECISION-PACKET.md` and naming the
  route finding (part (b) is `ContractD2h::OnTick`).
- `DOOR-DECISION-PACKET.md` item 6: the pointer to the new packet and the corrected route sentence.
- `research/INDEX.md`: the day-32 row. `STATE.md`: rewritten (under 15 lines).

## 4. Checks actually run

`bash tools/check-flags.sh` (`check-flags: no uncovered runtime names`), `bash tools/check-conflict-markers.sh`
(`OK`), `git diff --check` (clean, staged and unstaged), zero em dashes in every file this lane added or edited today
(`grep -c` of the character: 0 in each), a grep of the staged additions for provider names, hourly prices, ssh hosts
and 7-to-8-digit ids (only byte counts and dates matched), and the CPU gate of section 1. Every check gated in its own
`if ! ...; then exit; fi` line before each commit.

## 5. Observation for the lead (not acted on; a `docs/` registry is the lead's)

`docs/FLAGS.md`'s header section "Memory-shaped admission for a high session ceiling, 2026-09-09" (lines 3 to 12)
names a provider and a machine id for the pair the door was built for. The public-boundary policy's
`provider_machine_id` pattern is the gate for that form; whether that paragraph is grandfathered or should move to
darklanes is the lead's call. Not restated here.

## 6. Boundaries

No engine change; no `docs/` edit; no V4.1 code; no external dependency; no `--no-verify`; no third lock name; no GPU
run; no other lane's worktree or process touched (the lead's `wt-spill-integ41` seen only in `git worktree list`); no
host, id, location or cost in any tracked file; no cross-card comparison; no recommendation. Scratch:
`/tmp/spill-c-day32/` (the test log, copied into `day32/`, and the check logs) removed at close.
