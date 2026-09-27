# WP-A day 60: OWED item 11, Move 1's decision cell (i) on the current tree, both classes, same window

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. Every cell `executed-not-qualified`. Prepared while DAY59's 5090
cell waits for the card (text and scripts only; nothing here runs before DAY59's reading).

## 1. Pre-registration

**The item** (`OWNER-THREAD-OFFLOAD.md` Move 1 cells, day 16's clause, verbatim): "(i) The census stall cell repeated
with the second stream: the tenant's ITL max during the demote and promote arms against the day-16 baseline, N=5 per
arm per order, both orders, door ON in both ... Decision clause: `stall_median(second stream) <= idle p99` of the same
sitting for both classes". Lane C ran it on day 29 (`research/spill-c-20260919/DAY29.md`) on tree `653c997f4` against
the day-16 tree `1646d421b`: `DAY29 CELL(i) CLAUSE class=demote stall_median(second stream)=150.0 idle_p99_sitting=14.9
-> clause_not_met` (and the promote class the same way); ruling 47 carries the cell as owed on the current tree, the
clause read as written.

**The cell, unchanged but for arm X's tree.** C's day-29 scripts, verbatim (`research/spill-c-20260919/`:
`day29-stall-cell.sh`, `day29-box-run.sh`, `day29-stall-reading.py`): arm X the lane's current tip (the copy-stream
program with everything since, S4's receipts, V, the split lines), arm Y the day-16 tree `1646d421b` in its own
worktree (the owner-stream program); every boot the day-16 `on` boot; `stall_cell.py --mode demote` then `--mode
promote`, N=5 per arm per order inside each boot; order 1 X Y .. x5, order 2 Y X .. x5; one dry boot of Y first; one
RTX PRO 6000 Blackwell, one collector hold; the 27B NVFP4 MTP artifact.

**The rule, as written (no bound of mine).** Per class: `stall_median(arm X) <= idle p99 of the same sitting` is the
clause; the reader prints it met or not. The same-window pair `y_minus_x` per class and order is a reading.

**What it can say, stated now.** The clause compares the whole intruder's stall with the idle tenant: the demote and
promote intruders prime a fresh prompt on the tick (the day-16 cell's shape), so the clause is expected **not met** on
any tree while the intruder primes on the tick (day 29: 150.0 against 14.9); the pair then reads how much of the
class's own on-tick share the copy stream removed on the current tree. If the clause is not met, item 11 closes with
the reading, and whether a cell that isolates a class from its intruder's prime (Move 2's owed item 3, "a stall cell
that isolates each class") replaces it is the lead's call.

**What each card decides.** The target card (the day-16 clause's own rig class); the 5090 has no role.

**Budget.** 0.15 agent-day to prepare; about 40 minutes of a target card.

## 2. The sitting prepared (`pro-single-day60/`)

- `build.sh <tip> <receipts_root>`: arm X the tip in `/root/wt-a`, arm Y `1646d421b` in `/root/wt-a-day16`, each with
  its `build-{x,y}.log` and final `rc=` line (the shape C's runner waits for).
- `driver.sh <receipts_root> <model.gguf>`: C's `day29-box-run.sh` verbatim (the stall cell (i) and the hit gate, one
  collector hold each, bounded lock retries), then C's `day29-stall-reading.py` over `stall/ev`. About 40 minutes of
  card time; any host class with one RTX PRO 6000 Blackwell and the 27B artifact at
  `/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`.

## 3. The sitting, read as registered: CLAUSE NOT MET (as stated before it ran)

- Run by the lead on the DAY59 twin's box (one RTX PRO 6000 Blackwell Workstation card, a 16-core host), right after
  the twin, 03:20Z to 03:56Z. Arm X at `d4117fb2d` (binary sha256 `0486d059dd73f0ae..`, the DAY59 twin's same binary),
  arm Y at `1646d421b` (sha256 `17b583832f1c6ecc..`). The binaries stayed in the box's build trees; their hashes are
  in `stall/ev/binary-{x,y}.sha256` and the builds' `rc=0` in `build-{x,y}.log`. Mirror `pro-single-day60/box/`: 296
  receipts, sha256-checked against the box manifest. One dry boot of Y and 20 scored boots, every receipt admissible,
  every replay PASS, the tenant's text identical in every run. Start temperatures 33 C to 68 C.
- Verbatim (`box/reading-day60.log`):

      DAY29 CELL(i) class=demote order=o1 y_minus_x=+86.4 unc=0.3 -> isolated (Y owner stream 150.2 N=5; X copy stream 63.8 N=5)
      DAY29 CELL(i) class=demote order=o2 y_minus_x=+86.2 unc=0.3 -> isolated (Y owner stream 150.1 N=5; X copy stream 63.9 N=5)
      DAY29 CELL(i) CLAUSE class=demote stall_median(second stream)=63.8 idle_p99_sitting=12.9 -> clause_not_met
      DAY29 CELL(i) class=promote order=o1 y_minus_x=+76.0 unc=0.2 -> isolated (Y owner stream 138.8 N=5; X copy stream 62.9 N=5)
      DAY29 CELL(i) class=promote order=o2 y_minus_x=+76.0 unc=0.2 -> isolated (Y owner stream 138.8 N=5; X copy stream 62.9 N=5)
      DAY29 CELL(i) CLAUSE class=promote stall_median(second stream)=62.9 idle_p99_sitting=12.9 -> clause_not_met
      DAY29 CELL(i) CLAUSE: NOT MET (demote=False promote=False admissible=True); executed-not-qualified

  The hit gate beside it: `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)`, door OFF and ON.
- Read:
  - The clause is not met in either class, as section 1 stated before the cell. The intruder primes on the tick in both
    arms, and a prime is not a class's work.
  - The pair is the result. On the current tree the copy-stream program takes 86.3 ms (demote) and 76.0 ms (promote)
    off the tenant's stall, against the day-16 owner-stream program, in both orders (uncertainty 0.3 and 0.2). Day 29
    read X at 150.0 on `653c997f4`, so the current tree's X is 86 ms below day 29's X.
  - A reading beside it, not a clause (another cell, same binary, same box, an hour earlier): DAY59's twin read the
    single 72-word fresh prime (`prime-short`, the same intruder shape as `demote`) at 63.78 / 63.81 ms. X's demote
    arm at 63.8 / 63.9 is that prime to within 0.1 ms. So on this tree the demote class adds nothing measurable to the
    tenant's stall beyond its intruder's own prime. The promote arm's intruder is a different prompt, so it gets no
    such pairing here.
- **Item 11 closes with this reading.** Whether a class-isolating cell replaces (i) is the owner's decision. The
  candidate is registered as text in section 4, so the decision can be made on text.

## 4. Candidate, for the owner's decision: a cell that isolates each class from its intruder's prime

Not run and not scheduled. It replaces (i) only if the owner rules so, with the bounds as written here.

- **The cell.** One binary, the tip, and one sitting. Per class, the class arm is paired with a control arm whose
  intruder is byte-identical and does the same model work without the class's tier work:
  - demote: `demote` against the same 72-word fresh prompt with the prefix cache sized so that the insert evicts
    nothing (no demote is issued; the reader requires `server_demote_n=0` in the control and `> 0` in the arm);
  - promote: `promote` against the same `P_A` / `P_B` hits kept device-resident (a device restore instead of a
    promote; the reader requires `server_promote_n=0` in the control and `> 0` in the arm).
  - Interleaved: o1 is arm, control five times, and o2 is control, arm five times, both classes in each boot as in
    (i). Door ON, the 27B, one RTX PRO 6000 Blackwell, 250 ms telemetry.
- **The share.** Per class and order, `share = stall_median(arm) - stall_median(control)`.
- **The candidate clause.** `share <= idle_p99 - idle_p50` of the same sitting, for both classes and both orders:
  the class adds no more to the tenant than the idle tenant's own tick-to-tick spread (about 1.3 ms in this sitting).
- **What it would decide.** Met: Move 1's owner-thread goal is read as reached for that class on the current tree.
  Not met: the share is the class's remaining on-tick price, and it gets a design under its own pre-registration.
- **Budget, if ruled in.** 0.2 agent-day and about 45 minutes of a target card.
