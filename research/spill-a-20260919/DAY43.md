# WP-A day 43: OWED item 15, G''' against G4 at 4096-token entries (ruling 54)

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. Ruling 54 (`research/spill-lead-20260919/INTEGRATION-DAY12.md`):
"G''' against G4 at long entries is not an owner choice: pre-register item 15's 4096-token cell with both arms and run
it on the next target card; the evidence decides by your registered rule." Every cell `executed-not-qualified`.
Behind `MEMRA_KV_HOST_CONTRACTS` (default OFF).

## 1. Pre-registration (committed before any cell runs)

**What the record says.** Under G4 the D2H receipt kernel (`d2h_receipt_sha256`, one thread per item, a sequential
SHA-256 chain, about 40 MB/s per thread) runs on the copy stream ahead of the batch's copies: `SURVEY G items=32
bytes_each=4194304 N=5 ms median=107.120` on the 5090 (DAY38 section 2), and a 4096-token 27B entry is about 32 KV items
of 3.8 MB plus 48 recurrent spans (about 157 MB). Under G''' (`9ab5c1265`) it runs on a receipt stream beside the
copies. DAY38 section 16c named G'''s advantage as "a promote's copies never queue behind a demote's receipt kernel".

**A reading of the door's program, before the cell.** A promote Block-settles any pending demote before its own
submission (`host_demote_settle_pending(host, ContractWait::Block, "a promote")`, day 17's one-batch-in-flight rule),
and so do a second demote and a tenant purge; a hit on the Demoting entry parks until its publication (days 29 and
38). So in the door, under either arm, no promote's copies ever run beside a demote's receipt kernel: they wait for the
whole demote. What differs at serving shape is: (a) the demote's copy phase, which every path that waits on a demote
pays (G4: the kernel, then the copies, on one stream; G''': the copies beside the kernel, the landing at the later of the
two, so G''' is shorter by about the copies' time); (b) what runs beside the tenant's decode during the kernel (G''': two
side streams with work; G4: one); (c) D2D captures and restores queue behind the kernel under both (their digests
follow it on its stream in both). Section 16c's advantage holds for the engine alone, not for the door.

**Arms.** `g4` = the lane before design S2's code (`b4816eda8`, its crates equal to main `5d653e851`: G4 and T); `g3` =
the same crates with `item15/g3-placement.patch` (the crate diff `26676c037` to `9ab5c1265`, G4's placement change
reversed: the receipt stream back with the D2H receipt kernel and the D2D digests on it; applies cleanly, checked).
Both are built in the target sitting; `g4` is the S2 sitting's `g4` binary, copied with its hash.

**Harness** (`stall_cell.py`, two new modes; the five earlier arms byte-for-byte unchanged, their old receipts replay
PASS): `demote-long` (each timed intruder a fresh prompt of the prime arm's length, about 4096 tokens, whose insert
evicts the resident long entry into the host tier; one untimed long seed) and `promote-long` (two fixed long prompts
L_A and L_B seeded untimed; each timed intruder a CHAIN: a hit on L_A, which promotes and whose insert demotes L_B,
then at once a hit on L_B, which meets L_B Demoting, parks until its publication and promotes; the chained request's
e2e is `chain_wall_ms`, the path that waits on a demote). Readers: `item15-reading.py`, `item15-hump-reading.py` (day
38's hump program over `demote-long`).

**Environment** (the target card's PRO environment of the S2 sitting): the 27B NVFP4 MTP artifact, `MEMRA_SERVE_SPEC=0`,
`MEMRA_CTX=8192`, `MEMRA_MAX_SESSIONS=4`, `MEMRA_KV_HOST_MB=8192`, `MEMRA_KV_HOST_CONTRACTS=1`, and
`MEMRA_PREFIX_CACHE_MB=448` (one long entry, about 280 MB, fits; two do not).

**Cells** (`pro-single-i15/`, one RTX PRO 6000 Blackwell, the collector's hold, receipts under
`/root/spill-receipts/a-i15`, after the S2 sitting): `survey.sh` (the day-38 survey probe on this card: identity over 56
sizes and offsets, the kernel at 16 x 60 KiB, 32 x 60 KiB, 32 x 1 MiB, 32 x 4 MiB, N=5 each; a reading), `ab-long.sh
demote-long` and `ab-long.sh promote-long` (20 boots each, o1 = g4 g3 x5, o2 = g3 g4 x5, `--n 5`, each boot's start
temperature and SM clock recorded), `hump-long.sh` (xg4 xg3 xg3 xg4, `demote-long --n 8`, then both readers).

**Completeness** (`item15-reading.py`): every boot with a receipt, no error, every intruder and chain answered, one
replay PASS per boot, and in `demote-long` a copy-complete line for every timed run and the seed; two `xg3` hump boots.
An incomplete cell is not read and repeats whole once. If the budget holds two long entries or none (no demote per timed
run), the repeat sets `MEMRA_ITEM15_CACHE_MB` to 1.6 times the logged long entry's MB, rounded up to 32 MB, recorded.

**The rule, stated before any cell** (per order; medians over the pooled boots of an arm; steady = the second and later
such line of a boot). G''' is adopted for the RTX PRO 6000 class only if every term holds in both orders:

1. chain (promote-long's chained e2e): g4 minus g3 at least +2.0 ms;
2. copy (demote-long's steady `demote copy complete .. Xms from submission to completion`): g4 minus g3 at least +2.0 ms;
3. no regression: the tenant's e2e in each mode, g3 minus g4 at most +1.0 ms; the first intruder's e2e (promote-long),
   at most +1.0 ms; the steady demote wall (`wall Xms t0 to publication`, demote-long), at most +5.0 ms; the tenant's
   pre-fire ITL median in each mode, at most +0.05 ms;
4. the long-entry hump: g3's median HUMP (two boots, `item15-hump-reading.py`) at most 0.15 ms.

Otherwise G4 stays the single placement, and item 15 closes with the survey's price and the cell's readings recorded.

**What each card decides.** The target card decides the RTX PRO 6000 class only (one-rig evidence sets a one-rig
default at most). The 5090 keeps G4 whatever the target card reads, until item 16 places its hot-hold (f) FAIL. If the
rule adopts G''' for the PRO class, the per-card placement (G''' on the PRO 6000 class, G4 on the 5090, keyed on the
card class, the env seam for measurement) is designed and pre-registered next with its own acceptance; nothing ships
from this cell.

**Predictions.** (2) holds by about the copies' time (280 MB at the card's D2H rate, about 6 to 11 ms); (1) holds by
about the same; (3) holds; (4) holds (BOX7 read G''' flat at 64 tokens, +0.031). The kernel reads about 100 ms at 32 x
4 MiB on this card too.

**Budget.** 0.3 agent-day for the harness, the readers and the sitting (this commit); the sitting's run is about 3
hours of card time after the S2 sitting.

## 1a. Amendment before any reading runs: the completeness count and the entry length

- Reading one finished boot's server log to check the cell was demoting (`demote-long/ab/o1/b01-g4`, 10 copy-complete
  lines, no comparison read) showed section 1's completeness term mis-stated: "a copy-complete line for every timed run
  and the seed". The seed is demoted by timed run 1's insert, and the last timed run's entry stays resident, so a boot
  in which every timed run demoted carries exactly one line per timed run. `item15-reading.py` required runs + 1 and
  would have called every boot incomplete; it now requires one line per timed run. No rule term changes.
- The long prompt (`PRIME_TARGET_TOKENS - 4` words) reads `intruder_prompt_tokens=[5122, ..]`: the entries are about
  5.1k tokens, not 4096; the cell is read as it ran.
- The box's run of the sitting calls the reader from its tree at `c62a34175` (the defect included); the reading is
  re-run with the corrected reader over the same raw receipts once they are mirrored, and both outputs are banked.
