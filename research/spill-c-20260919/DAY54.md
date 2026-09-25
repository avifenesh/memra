# WP-C day 54 (2026-09-24): the double-park slice, pre-registered (OWED C4)

`OWED.md` C4. Source: `DOOR-DECISION-PACKET.md` section 5 item 3 and section 6, `DAY29.md` Task 2's finding: on the
integ38 tree the promote intruder's hit parks twice (the promote, then Move 2's restore off the tick) and its inline
demote publishes `after 1 poll(s)` in a median 22.4 ms where the day-23 tree (`crates/` = main `0713c1a79`) read
97.0, so the demote's two hashes land on one tick and the promote arm's tenant stall reads 149.6 where the day-23
tree read 81.9. "Which slice between `0713c1a79` and the integ38 tip moved the copy's landing is not determined."
Written before any slice binary was built or booted; tree at start: `6f7496650`.

## 0. What the RTX 5090 already shows, and why the cell is on the target card

The 5090's day-37 cell ran the promote class on a tree past the integ38 tip (`091a931c0` contains `643ecbb28`;
`rtx5090-day37/stall/ev/p1-base/pass1/on/server.log`, the 9B): 10 `promote submitted off the tick` lines, 0
`restore submitted off the tick` lines, and `[prefix-cache] restore not routed (contracts door): the entry was
promoted for this admission (insertion pin ...); the tick program copies it`, the promoted-pin refusal of A day 26
(ruling 36's proposal 1, `e008bf502`), which removed the second park after integ38. So no 5090 receipt exists at
the slice trees, and whether this card's shape would show the landing move is unknown. Corrected before any run
(the first version of this section read the 5090's missing park as a property of the card; it is a property of the
later tree). The cell runs on the target card, where the finding was made, with day 29's shape and the slice trees
themselves (all before A day 26), not a 5090 reproduction first: the question is the target card's.

## 1. Pre-registration

**The slices.** The first-parent commits of `main` from `0713c1a79` to the integ38 tip `643ecbb28` are 16; nine
change nothing under `crates/`, `Cargo.toml` or `Cargo.lock` (docs, data, merges of lane records), and two change
only a test (`cc754b476`) or the docs.rs build stub (`9d4b17761`). The binaries are the base and the seven
runtime-changing slices, each built at its commit:

| label | commit | what it merged |
|---|---|---|
| `e0` | `0713c1a79` | the day-23 tree's crates |
| `s1` | `ff64e7f5d` | #624, request fault boundary |
| `s2` | `da1f59bf6` | #631, integ33 |
| `s3` | `226abab0e` | #632, integ34 |
| `s4` | `5df11152f` | #634, integ36 |
| `s5` | `58b814abe` | #638, integ37 (the parked-only wait) |
| `s6` | `f661406e4` | lane A into integ38 (A day 22, the D2D receipt term) |
| `s7` | `269ef2cec` | the Block arm's host wait on the D2D receipt event (the integ38 tip's runtime) |

**The cell `slices`** (`day54-slice-cell.sh` through `day54-box-cell.sh` and the day-40 runner; one collector hold,
`--rig pro-single`, `/tmp/memra-gpu.lock`, 250 ms telemetry; the 27B `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`). Every boot is
day 29's: lane A's day-16 script's `on` boot (`MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 MEMRA_SERVE_SPEC=0
MEMRA_PREFIX_CACHE_MB=256 MEMRA_KV_HOST_MB=8192 MEMRA_KV_HOST_CONTRACTS=1`) running lane A's harness
`research/spill-a-20260919/stall_cell.py` byte for byte (its SHA-256 recorded), `--mode promote` only, N=5 per arm per
order inside the boot. One dry boot of `e0` first (`--n 1`, part of no quantity; if it fails the cell stops and the
failure is quoted, no patch). Then order 1 = `e0 s1 ... s7` five times and order 2 = `s7 ... e0` five times: ten
boots per binary, eighty boots, about 85 minutes (day 29's boot, ready and promote half took about 62 s).

**Admissibility (a failure voids the reading).** 80 receipts, every one `STALL REPLAY: PASS`; per binary ten boots,
no run errors, one tenant text.

**Readings** (`day54-slice-reading.py`, every constant fixed here). Per binary over its 100 promote-arm runs, from
each run's `server_log_lines`: the landing, `demote published off the tick: ... Xms from submission to completion`
(median, p25, p75, and the runs with no such line); the parks per run (`promote submitted off the tick`, `restore
submitted off the tick`, medians); the tenant's stall (the median over its ten boots of each boot's promote-arm
`stall_ms` median).

**The rule.**
- Endpoints: `reproduced` iff `landing(e0) > 2 x landing(s7)` or the restore parks per run differ between `e0` and
  `s7`; otherwise `not_reproduced` (the finding does not recur on this sitting's box, reported as such, and no
  slice is named).
- A binary's landing class is `late` above the midpoint of the endpoints' medians and `early` at or below it. The
  moving slices are every consecutive pair in first-parent order whose class or restore-park count differs;
  verdict `moved_at <labels>` (more than one label is a result: the landing moved in steps), `no_single_slice` if
  the endpoints differ and no consecutive pair does (impossible by construction unless a median is missing).

**What this decides.** Nothing about the door by itself: it names the slice for the contracts door's review
(2026-10-05), as `DOOR-DECISION-PACKET.md` item 3 asks. The reader was dry-checked on day 29's arm-X receipts
relabeled as all eight binaries (`landing 22.4` everywhere, `not_reproduced`, as it must read).

## 2. Results, cell `slices` (the target card, BOX8; receipts `pro-single-day52/slices/`)

One collector hold, 23:37:38Z (the dry boot of `e0`) to 00:47:48Z, 81 boots, the 27B, lane A's harness
`13867e77...`, tree `f9f5f3953`; the eight servers built on the box from their commits (`builds.log`). Regime
(`slices/regime.log`, the collector's 250 ms CSV, N=16778): SM 2610 to 2865 MHz, power 89.9 to 496.8 W, 47 to 72 C.
81 of 81 receipts `STALL REPLAY: PASS`; collector `--validate` rc=0.

Verbatim (`slices/reading.log`):

`DAY54 CHECKS receipts=80 replays_pass=81 admissible=True e0:boots=10,errors=0,texts=1 s1:boots=10,errors=0,texts=1 s2:boots=10,errors=0,texts=1 s3:boots=10,errors=0,texts=1 s4:boots=10,errors=0,texts=1 s5:boots=10,errors=0,texts=1 s6:boots=10,errors=0,texts=1 s7:boots=10,errors=0,texts=1`

`DAY54 SLICE label=e0 commit=0713c1a79 boots=10 promote_runs=100 landing_ms median=89.1 p25=88.8 p75=89.6 n=100 landing_missing=0 promote_parks_per_run median=1.0 restore_parks_per_run median=0.0 stall_median_of_boots=75.7`

`DAY54 SLICE label=s1 commit=ff64e7f5d boots=10 promote_runs=100 landing_ms median=89.1 p25=88.7 p75=89.8 n=100 landing_missing=0 promote_parks_per_run median=1.0 restore_parks_per_run median=0.0 stall_median_of_boots=75.6`

`DAY54 SLICE label=s2 commit=da1f59bf6 boots=10 promote_runs=100 landing_ms median=89.1 p25=88.8 p75=89.6 n=100 landing_missing=0 promote_parks_per_run median=1.0 restore_parks_per_run median=0.0 stall_median_of_boots=75.7`

`DAY54 SLICE label=s3 commit=226abab0e boots=10 promote_runs=100 landing_ms median=89.1 p25=88.8 p75=89.4 n=100 landing_missing=0 promote_parks_per_run median=1.0 restore_parks_per_run median=0.0 stall_median_of_boots=75.7`

`DAY54 SLICE label=s4 commit=5df11152f boots=10 promote_runs=100 landing_ms median=89.1 p25=88.8 p75=89.6 n=100 landing_missing=0 promote_parks_per_run median=1.0 restore_parks_per_run median=0.0 stall_median_of_boots=75.7`

`DAY54 SLICE label=s5 commit=58b814abe boots=10 promote_runs=100 landing_ms median=26.2 p25=26.0 p75=26.7 n=100 landing_missing=0 promote_parks_per_run median=1.0 restore_parks_per_run median=1.0 stall_median_of_boots=98.9`

`DAY54 SLICE label=s6 commit=f661406e4 boots=10 promote_runs=100 landing_ms median=26.3 p25=26.1 p75=27.0 n=100 landing_missing=0 promote_parks_per_run median=1.0 restore_parks_per_run median=1.0 stall_median_of_boots=98.8`

`DAY54 SLICE label=s7 commit=269ef2cec boots=10 promote_runs=100 landing_ms median=26.3 p25=26.2 p75=27.0 n=100 landing_missing=0 promote_parks_per_run median=1.0 restore_parks_per_run median=1.0 stall_median_of_boots=98.8`

`DAY54 ENDPOINTS e0 landing=89.1 restore=0.0 s7 landing=26.3 restore=1.0 rule (e0 landing > 2 x s7 landing) or (restore parks differ) -> reproduced`

`DAY54 MOVES midpoint_ms=57.7 s5=58b814abe: landing late->early (89.1->26.2 ms), restore parks 0.0->1.0`

`DAY54 VERDICT -> moved_at s5=58b814abe`

**Read, not tuned.** The double park and the moved landing arrive together in one slice, #638 (integ37, the
parked-only wait): the restore of the promoted entry starts parking off the tick (0 to 1 per run), the inline
demote's copy then lands 89.1 to 26.2 ms after submission, and the tenant's promote-class stall rises 75.7 to 98.9
ms. No other slice moves either reading. For `DOOR-DECISION-PACKET.md` section 5 item 3 (the question it asked is
answered here; the packet is updated with the verbatim lines when C3 reads this sitting in).
