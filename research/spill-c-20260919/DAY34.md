# Session C day 34: the door packet and the door doc re-read against A days 26 and 27 and C day 33

Lane `lane/spill-c-20260919`, checkout `wt-spill-c`, start `ab7ef9f18` = remote, clean. No card work today; the one
CPU-heavy step (regenerating A's double-park lines from their raw receipts) ran under `systemd-run --user --scope -p
CPUQuota=1200% -p MemoryMax=28G`. Every push in the announced `MEMRA_RELEASE_QUALIFICATION_MODE=development` mode (the
hook prints `UNQUALIFIED DEVELOPMENT: refs/heads/lane/spill-c-20260919 at <sha>; no GPU qualification claimed` and
records the skip in the clone's `.git/memra-gate-skips.log`). No engine change, no `docs/` registry edit, no commit on
main, no PR, no recommendation anywhere in the two documents.

## Merges (first action)

- `f449151c9`: `origin/main` at `22f7489a3` (after #647, the integ42 PR) merged `--no-ff`, clean.
- `67d209b10`: the local ref `lane/spill-integ43-20260922` at `7cd2fc95d` (no PR, not on origin) merged `--no-ff`; at
  that SHA the ref was main plus this lane, so the merge carried no file delta. Pushed.
- `b64647abb`: the same ref had moved to `d06a9dd8c` (the lead's integ43 record opened: A day 27 `8bfd3d103` merged,
  `HOSTPREFIX-DOOR.md` section D item 6 resolved three-way with A's code paragraph and C's day-33 paragraph both kept,
  ruling 39). Merged `--no-ff`, clean (no conflict, so no side of `HOSTPREFIX-DOOR.md` had to be chosen; the integ
  side is the tree). Pushed. This is the tree the day's edits sit on.

## 1. What was read, and what each reading changed

**A day 27 section 1 (`research/spill-a-20260919/DAY27.md`, file:line on `cb9fa5ef3`; integ43).** The demote's
`in - completion` (74.8 ms on the target card, 21 to 23 ms on the RTX 5090) is ONE SHA-256 pass of `bind_tier_image`'s
`StateBundle` checksum over the whole host image (`worker.rs:9344` inside `add`, from `host_demote_publish` `12258` /
`12265`, on the worker thread inside the tick), about 157 MB of it recurrent f32 in pageable heap (`HostF32::Heap`,
`8261-8268`, `8594-8596`) that never crossed the contract and has no receipt partner; OFF never computes it (`hpx.tier`
`None` unless `kv_host_contracts && hpx.budget > 0`, `20963-20978`; the early return `9298-9300`). Move 1's two receipt
hashes (`tier_transfer.rs:1710-1712` at the poll, `9440-9451` at bind) cover the KV planes only, about 1.9 MB on the 27B,
about 0.9 ms each. The packet's section 2 had said "the Move 1 demote's two host hashes stay on the owner thread inside
the tick ... the hash micro-cell below prices one pass"; that sentence was replaced by the attribution with those
pointers, and the micro-cell rows are now described as pricing a whole-image pass, not the two receipt hashes.

**A day 27 sections 2 and 3.** The three options (a helper thread for the heap payloads; the slice-3 four-lane program
for the receipt term, (b) and (b'); a copy-stream digest over PCIe, unmeasured) and the digest micro-cell per host:
target host `sha_ms=77.589 lanes_ms=70.737 lanes_over_sha=0.912`, 5090 host `sha_ms=37.111 lanes_ms=55.078
lanes_over_sha=1.484`, both read from `ev/digest-micro.log` (regimes from each cell's `command.gpu.csv`: 33 C and 33.50 W
over 8 samples; 56 C and 16.54 W over 5). The baseline for option (a) (the day-26 cell on the day-27 tree): stall
`-3.4 / -3.5 unc=0.1 isolated` (81.9 / 81.8 against 85.2 / 85.3), e2e `+91.3 / +91.3 isolated` (206.6 / 206.7 against
115.4 / 115.5), `demote_in-completion median=74.8`, gaps 95.3 and 92.4; 33 to 51 C, 33.27 to 359.98 W (4671 samples).
Both went into the packet's section 4 (target table: two rows; 5090 table: one row) and the door doc's section B (three
rows plus an attribution row).

**A day 26 (`DAY26.md`, on `main` since #647).** The double park after the promoted-pin refusal: stall 81.8 / 81.8
against 85.3 / 85.3 (`on_minus_off=-3.4 unc=0.1 -> isolated`, both orders), e2e 206.8 / 206.7 against 115.3 / 115.4
(`+91.4 unc=1.2 / 1.1 -> isolated`), `demote_in-completion median=74.8` (74.9 before), `demote_in median=172.3` (97.2
before), gaps 95.3 and 92.4, 100 of 100 ON runs `parked_per_run=[1] restore_submitted_per_run=[0] not_routed_per_run=[1]`;
32 to 52 C, 31.98 to 360.81 W (4672 samples). The packet's section 4 gains the row, item 3 gains the result and ruling
37. The door doc's section B already carried A day 26's row (from the integ42 merge), so it was not added again.

**A day 26's lines were regenerated, not copied.** `pro-single-day26/box/` banks no reading log, so
`day26-reading.py pro-single-day26/box` and `day25-double-park-reading.py pro-single-day26/box/double-park/ev` were run
over the raw `ev/` (under the CPU quota): every line above came out of those runs and equals `DAY26.md`'s. Day 27's
regenerated lines equal the banked `pro-single-day27/box/reading-day25.log` and `reading-day26.log`.

**C day 33 (`DAY33.md`).** The `DAY33 HASH-WC VERDICT` line went into the packet's 5090 table verbatim (`H1 refuted`;
the WC rate size-independent at 0.116 GB/s; the fastest WC pass 7.8x the slowest ON demote). The day-33 paragraph's
tail in the door doc's item 6 ("A day 27 reads it and its answer lands separately ... untouched by day 33") now closes
on A's paragraph above it (H2's form, heap) and points the promote side at A's item 5.

**Integ43 (`research/spill-lead-20260919/INTEGRATION-DAY12.md`, on the lead's ref).** Ruling 39: "option (a) is
approved for A day 28 as specified", with the receipt term unchanged, `Hashing` as a `PendingDemote` state, one
long-lived helper per `HostTierContext` joined at shutdown and `host.disable`, the two fault values under the existing
`MEMRA_KV_HOST_FAULT` row, a bitwise digest unit cell. The lead's reading of the 32 ms against the 74.8 is quoted in the
packet's item 2 verbatim ("this pass minus what OFF spends on the same tick") and not re-derived.

## 2. The packet (`DOOR-DECISION-PACKET.md`), paragraph by paragraph

- Status header: a day-34 sentence (what is on `main`, what is on the integ43 ref, what was re-read).
- Section 2, "what still runs on the tick under ON": the attribution replaces the two-hashes sentence (above).
- Section 4, target table: A day 26's row, the day-27 baseline row, the digest micro-cell row (target host). 5090
  table: the day-33 verdict row and the digest micro-cell row (5090 host). Every figure with its N, order, regime
  and receipt path; no cross-card figure.
- Item 2: the "about 32 ms over OFF ... not attributed by any cell" clause is now attributed (the bundle checksum),
  the lead's reading quoted, and option (a) named as the arm with its acceptance gate stated as a gate (the five
  clauses of `DAY27.md` section 2 plus ruling 39's additions), with the baseline it moves from and the statement
  that its receipt does not exist.
- Item 3: A day 26's result and ruling 37 appended; "which slice moved the copy's landing" stays not determined.
- Item 7: items 5, 6 and 9 dropped as resolved (with where each was resolved); what stays: the arena handoff, the
  DFlash tail slice, verify digest v3, the 9B's KV byte split, option (a) itself, a tenant-stall cell on the 5090.
  The promote-side census question is closed by A's code census and named as such.
- Section 6: the naked-default bullet says a default on either class would carry the bundle checksum on the tick
  until option (a) lands and passes its gate; the longer-door bullet's candidates are re-listed (option (a)'s gate,
  a 5090 tenant-stall cell, the double-park slice, the KV byte split; the 5090 pair and the retire-settle price
  removed as done).
- Appendix A: how the day-34 numbers were traced (the regenerations, the rule lines, the CSVs).

## 3. The door doc (`HOSTPREFIX-DOOR.md`)

- Section B: the day-27 baseline row, the attribution row (0 under OFF against one pass under ON, with the pointers),
  the digest micro-cell rows per host. A day 26's row was already present; not duplicated.
- Section D: item 6 marked RESOLVED day 27 and day 33 with both pointers and the one unmeasured term (the 9B's KV
  byte split); items 5 and 9 already read RESOLVED day 31 with their pointers (unchanged).
- Section E: one DAY 34 paragraph, what changed since the packet's first draft; the question itself unchanged and
  still not answered here.

## 4. Records and checks

`DAY34.md` (this file), `STATE.md`, `research/INDEX.md` row `spill-c-20260919/day34`. Checks at close:
`tools/check-flags.sh` (`no uncovered runtime names`), `tools/check-conflict-markers.sh` OK, `git diff --check` clean,
zero em dashes in every line added today, no provider host, id, price or location in the added lines,
`python3 tools/check-public-boundary.py check`: `599 matches (599 grandfathered, 0 new)`. Boundaries: no engine change,
no `docs/` registry edit, no V4.1 code, no external dependency, no `--no-verify`, no GPU run, no touch of other lanes'
worktrees or processes (the lead's integ43 ref was read as a ref only), no cross-card comparison, no recommendation.
Scratch: `/tmp/spill-c-day34-checkflags.log` removed at close. Budget: about 1.7 agent-hours against 3.

## Push section

- `67d209b10` (the main merge plus the first integ43 ref) and `b64647abb` (the moved integ43 ref), each
  `UNQUALIFIED DEVELOPMENT: refs/heads/lane/spill-c-20260919 at <sha>; no GPU qualification claimed`,
  `pre-push: skip recorded`.
- `3ce00ed36`: the packet (cell 1), the same two hook lines.
- `fba59c2b5`: the door doc (cell 2), the same two hook lines.
- The records tip: this file, `STATE.md`, the INDEX row; its SHA is the commit that carries this line.
