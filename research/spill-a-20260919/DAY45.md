# WP-A day 45: OWED item 16, the 5090 hump replicate in G4's hot regime (ruling 54)

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. Ruling 54: "the 5090 (f) FAIL for G4 stands as registered ... the
FAIL's cause is unplaced between the card's thermal regime and the design. Owed: section 20's cell replicated in the G4
hold's thermal regime (drive the card to that regime before the boots, record the clock and temperature per boot),
pre-registered before it runs, with a rule that places the cause either way." Every cell `executed-not-qualified`.

## 1. Pre-registration (committed before any cell runs)

**The two holds on record.** DAY38 section 19 (the G4 hold): its hump cell at 87 to 88 C with the SM clock falling from
1995 to 1830 to 1970 MHz at 150 to 164 W, after the hold's demote A/B at 86 to 88 C; `HUMP arm=xg4 boots=2
median-hump=+0.299` (FAIL against 0.15), the control `xgpp .. +0.601`. DAY38 section 20a (the base-controlled cell, its
own hold, no warm-up): 60 to 78 C; `xbase +0.072`, `xg4 +0.032`, `xg3 +0.055`, `xgpp +0.334`.

**The cell** (`rtx5090-day45/hot-hump-run.sh`, one bounded hold of `/tmp/memra-5090.lock`; binaries by
`rtx5090-day45/build.sh`: base `80039a8de`, g4 `26676c037` (section 19's g4 crates), g3 `9ab5c1265`, gpp `358749c9f`):

1. **The warm-up is the G4 hold's own step before its hump cell**: section 3's demote A/B, base against g4, o1 = base
   g4 x5, o2 = g4 base x5, `stall_cell.py --mode demote --n 5`, the day-35 5090 environment. A reading only; it drives
   the card to that hold's regime the way that hold did.
2. **Section 20's eight boots**, unchanged: `xbase xg4 xg3 xgpp xgpp xg3 xg4 xbase`, each `--mode demote --n 8` (16
   demote runs), each boot's start temperature, SM clock and power and its local start and end stamps in `BOOT.txt`; the
   250 ms telemetry throughout the hold.
3. `day38-hump-reading.py` (each arm's median HUMP over its two boots), then `item16-reading.py`.

**The regime check, stated now.** The hot regime is reproduced if every hump boot starts at 85 C or above and its SM
clock median over its own window is at most 2000 MHz (section 19's hump cell). If it is not: `REGIME NOT REPRODUCED`,
nothing is placed, and the cell repeats once in a new hold with the warm-up doubled (the promote A/B of the same arms
added, 40 boots); a second miss is recorded and goes to the lead with both holds' telemetry. The control must hump
(xgpp's median above 0.15 ms) and every arm must read over two boots, or the cell repeats once.

**The placing rule, stated now** (r = xbase's median HUMP; d = xg4's median HUMP minus r):

- d above 0.15 ms: the cause is **the design** (G4 rises beyond the base arm in the regime that failed it). G4's 5090 (f)
  FAIL is placed on G4; a 5090 revision (or the per-card placement) is pre-registered next.
- else xg4's median HUMP above 0.15 ms: the cause is **the regime** (the base arm, with no device receipt at all, shares
  the rise within the bound). The FAIL is placed on the card's thermal regime; (f) as registered stays FAIL, and the
  reading `d` goes to the lead with it.
- else: **not reproduced** (G4 is flat in the regime that failed it; section 19's FAIL is one hold against two flat
  ones). Recorded as it reads; the lead reads G4's 5090 (f) from the three holds.

G''' (xg3, its own section-16 FAIL +0.422 in a hot hold) is placed by the same rule, a reading.

**Predictions.** The regime reproduces (the warm-up is the hold that produced it). The base arm rises (DAY38 section 19
read it rising in two of four hot A/B cells): the regime.

**Budget.** 0.2 agent-day to prepare (this commit); about 45 minutes of the 5090 once it is back from its reset.
