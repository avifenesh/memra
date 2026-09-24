# WP-C day 54 (2026-09-24): the double-park slice, pre-registered (OWED C4)

`OWED.md` C4. Source: `DOOR-DECISION-PACKET.md` section 5 item 3 and section 6, `DAY29.md` Task 2's finding: on the
integ38 tree the promote intruder's hit parks twice (the promote, then Move 2's restore off the tick) and its inline
demote publishes `after 1 poll(s)` in a median 22.4 ms where the day-23 tree (`crates/` = main `0713c1a79`) read
97.0, so the demote's two hashes land on one tick and the promote arm's tenant stall reads 149.6 where the day-23
tree read 81.9. "Which slice between `0713c1a79` and the integ38 tip moved the copy's landing is not determined."
Written before any slice binary was built or booted; tree at start: `6f7496650`.

## 0. What the RTX 5090 already shows, and why the cell is on the target card

The 5090's day-37 cell ran the promote class on a tree past the integ38 tip (`rtx5090-day37/stall/ev/p1-base/pass1/
on/server.log`, the 9B): 10 `promote submitted off the tick` lines and 0 `restore submitted off the tick` lines, the
inline demote publishing in 32.3 to 77.5 ms. The double park does not occur on this card's shape, so the 5090
cannot place the slice; the cell runs where the finding was made, on the target card with day 29's shape.

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
