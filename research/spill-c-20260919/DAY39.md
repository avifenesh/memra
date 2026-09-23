# WP-C day 39 (2026-09-23): the target card's demote-class tenant-stall cell

Lane `lane/spill-c-20260919`. Start tip `18c900fc7`; `origin/lane/spill-integ47-20260923` (`160929a92`, A day 30
and C day 38) merged as `a20e4990d`; the packet status fix is `d036db237` (A day 30 landed on integ47, not on
`origin/main` `9717e8d57` at that time; `day39-cpu/ancestry.log`). `origin/main` took integ47 during the hold: #656
(`5f1b0eda4`) merged at 22:01:33Z, and the post-run re-check in the same log reads `160929a92` an ancestor of it; the
packet's "not on main" sites now read "on `main` since #656". Every cell here is `executed-not-qualified` development
evidence. No number here is compared across cards or boxes; the RTX 5090 figures of days 35, 37 and 38 are context
only and never a denominator. No recommendation.

## 0. The binaries, recorded before any build

Each binary is built on the target card's box from the SHA below, in this lane's own box worktree, by
`day39-box-build.sh` (a detached checkout, a clean-tree check, `cargo build --release -p memra-server`, a copy of the
binary with its SHA-256 and the tree SHA it came from). The SHAs were fixed by the day-39 brief and are recorded here
before the first build starts.

| label | SHA | what the engine is |
|---|---|---|
| B0 | `091a931c0` | day 37's base tree: day 35's tree, main after #648 plus the integ43 ref, the pre-option-(a) engine |
| B1 | `9717e8d57` | `origin/main` at the day-39 fetch: option (a) (A day 28), option 2a (A day 29) and the bounded latch close (#652), without the D2H spans |
| B2 | `160929a92` | `origin/lane/spill-integ47-20260923` at the day-39 fetch: B1 plus A day 30's D2H spans (`97a9e091f`, `fc637d26a`); `git diff 9717e8d57 160929a92 -- crates` is those two commits' seven files and nothing else, and this lane's tip carries no crate diff against it |

## 1. Pre-registration (committed and pushed before the first boot)

**What is owed.** `DAY37.md` section 7 and `DAY38.md` section 7: "The target card's demote-class tenant-stall cell on
the option (a) tree: no cell has run (the packet's item 2 scope, item 7 and section 6)." Today's cell is that cell,
on one RTX PRO 6000 Blackwell (96 GB, 600 W), collector rig `pro-single`, lock `/tmp/memra-gpu.lock`.

**Unchanged from day 37.** The harness `day35_stall_cell.py` byte-for-byte (SHA-256
`9c9b38782db8798d16fc9189bdeda9c4fe4742251584abedd55a261615ae0ccc`). The five arms (prime, demote OFF, demote ON,
promote OFF, promote ON), the tenant fired at its 24th token, the prime, demote and promote intruders, N=5 per arm per
order with both orders inside every harness run, the boot env (`MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4
MEMRA_SERVE_SPEC=0 MEMRA_COMPAT=openai`, the prime boot `MEMRA_PREFIX_CACHE_MB=0 MEMRA_KV_HOST_MB=0`, the ON boots
`MEMRA_KV_HOST_CONTRACTS=1`), the per-run landing rule (`fired_at_ms + wall_ms <= tenant_wall_ms`), the BAD-line
list, the promote shape (1,1,1,1,0), the bounded card-free wait and the lock-refusal retry (15 waits of 120 s), the
program layout (pass 1 prime, off, on; pass 2 on, off, prime), the rule, the four outcomes, the DiD, the quantities
(stall, `top1_plus_top2`) and day 38's per-tick reader.

**What changed, all of it the rig and the third binary.**

- The rig: `/tmp/memra-gpu.lock`, `--rig pro-single`, the target card's shape from A day 16 (`MEMRA_PREFIX_CACHE_MB=256
  MEMRA_KV_HOST_MB=8192`, `--tenant-max-tokens 160`), port 18132 (day 23's), the 27B artifact
  `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf` checked against its recorded SHA-256 before the hold (`stall/provenance.log`).
- Three binaries (section 0), built on the box by `day39-box-build.sh`, each a copy with its tree SHA and SHA-256 in
  `builds.log`; the cell refuses to start if a copy no longer hashes to its build receipt.
- Six programs in ONE collector hold, in this order: **p1-b0, p2-b1, p3-b2, p4-b2, p5-b1, p6-b0** (ABCCBA). Each
  program is six boots, 36 boots in the hold.
- The scripts are day 37's and day 38's under day-39 names: `day39-stall-cell.sh`, `day39-box-run.sh` (collector
  timeout 5400 s), `day39-stall-reading.py`, `day39-tick-split.py`, `day39-regime.py`. Diffs against the day-37 and
  day-38 sources are banked in `day39-cpu/diffs/{cell,runner,reader,tick,regime}.diff`.

| script | SHA-256 at pre-registration |
|---|---|
| `day39-stall-cell.sh` | `290c17611dfe87992055b7e37944decc2c1525b52f1b0ba12af8e8e4d1e68f9e` |
| `day39-box-run.sh` | `280b3c8ac731cf62265df5732bbeccb13bf5350fef95ef0aebbe724af2104817` |
| `day39-stall-reading.py` | `83ae72f8c82ca047ca6c9d3f0ad3eb0cec900332f45626ab953c92a4137ab2fc` |
| `day39-tick-split.py` | `e8cb808b6330259247a041142c66ac66ee1d03765734af832b5be8393065806e` |
| `day39-regime.py` | `7734d3d5fa6b11ce5ef6fd3a4e9a88f1c7986f8dc79e14316d79106803dc4c0f` |

**N per program.** Per receipt N=10 (5 per order). Per program per arm N=20 (its two passes pooled). Per binary per arm
N=40 (its two programs). Per block side N=20. 60 receipts in the hold (6 programs x 10). A receipt with fewer landed
runs is `inadmissible` under day 35's rule; no receipt is re-run inside the hold and none is dropped from a pool.

**The budget decision, made before any run: three binaries fit.** A day 16's target-card boots took 68.2 s (prime),
101.3 s (off) and 111.0 s (on) boot to stopped; one program estimates 561 s, six programs 56.1 min, 64.5 min with 15
percent per-boot overhead (`day39-cpu/wall-estimate.log`). That is under the 90-minute hold limit and inside today's
4-hour budget, so B2 is run, not stated as owed. The collector's 5400 s timeout ends the hold at 90 minutes if it runs
long; programs not finished by then are reported as missing and their arms `inadmissible`.

**Admissibility per receipt.** Day 37's, with the ON demote line set per tree:

- b0: day 37's base set: every ON demote run with a demote line has `demote submitted off the tick`, a `D2H receipt
  ... require=ok` and `demote published off the tick`.
- b1: day 37's option (a) set: `demote submitted off the tick`, a `D2H receipt ... require=ok`, `demote copy complete
  off the tick` and `demote digests landed off the tick`, and zero `demote published off the tick`.
- b2: the b1 set, and every `demote copy complete off the tick` line of the run carries an `items=N (K KV, S f32
  spans)` term with S >= 1. A day 30's worker.rs prints it on every copy-complete line (`items=128 (32 KV planes, 96
  f32 spans)` on the 27B, `A/DAY30.md`); a B2 run without it is not running A day 30's program.

**The contrasts and the rule.** Day 37's rule, per contrast, per arm, per order block:

- contrast b1-b0 (the owed row): o1 = p2-b1 against p1-b0 (B1 after B0), o2 = p5-b1 against p6-b0 (B1 before B0).
- contrast b2-b1: o1 = p3-b2 against p2-b1 (B2 after B1), o2 = p4-b2 against p5-b1 (B2 before B1).

d = median(new) minus median(old), each side the program's two passes pooled (N=20); unc = the quadrature of the two
IQRs; `isolated` when |d| > unc. An arm has **moved** when both blocks are isolated with the same sign;
`order_split` when both are isolated with opposite signs; `under_resolution` otherwise; `inadmissible` when any
receipt of either block is. The DiD per class per block is (ON minus OFF)_new minus (ON minus OFF)_old, unc the
quadrature of the four IQRs, the same four outcomes. The primary lines are the demote-on arm and the demote DiD of
each contrast. The prime and OFF arms are controls for the part of the tree difference that is not the door. Both
quantities are read under this rule: `stall` (the harness's per-run worst ITL minus p50) and `top1_plus_top2` (the
tenant's two largest ITL gaps per run, summed).

**The tick reader (day 38's, unchanged).** Tick 1 is the first gap greater than 3 x p50 at index >= fire_at - 1; tick
2 is the next such gap; a quantity is `not_defined` for an arm when any run of it lacks the tick. The same rule and
DiD per contrast on q=tick1 and q=tick2. The reader runs only on a reading that printed `DAY39 ADMISSIBLE: 60 of 60
receipts; all=True`; otherwise it prints `DAY39 TICK REFUSED`.

**The hypothesis (day 38's H, P1 to P5 unchanged), on contrast b1-b0 only.** H: option (a) moves the bind pass off
the poll tick (tick 2) and leaves the insert's tick (tick 1) alone. P1: demote-on q=tick2 `moved` with d < 0 in both
blocks. P2: demote-on q=tick1 `under_resolution`. P3: did-demote q=tick2 `moved` with d < 0 in both blocks. P4:
did-demote q=tick1 `under_resolution`. P5 (control): demote-off `under_resolution` on both ticks. Refutation and the
P5-fails reading are day 38's. Contrast b2-b1 carries **no** prediction: the D2H spans change what the ticket carries,
and no tick is predicted for it; its lines are read under the rule and described.

**Also read, described, not ruled on.** The option (a) ledger per tree (b1, b2) in day 36's form, b0's count of
ledger lines (expected 0), day 35's attribution per tree, and for b2 the `items=` terms of the copy-complete lines.

**Telemetry and the thermal regime.** The collector samples the card at 250 ms (`command.gpu.csv`: pstate, SM and
memory clocks, power draw and limit, temperature, memory used, utilization, PCIe link) across the whole hold; the
cell's own 1 s sampler writes `ev/card.during.csv` beside it. `day39-regime.py` prints the hold, the marks window and
each program's window (first `boot-<prog>-` mark to last `stopped-<prog>-` mark): samples, temperature, power draw and
SM clock ranges with medians, and the memory range. The box's UTC offset is recorded on the box
(`host_utc_offset=` in `stall/provenance.log`) and passed to the reader, because nvidia-smi stamps local time. The
regime is described per program, not used by the rule; a program whose regime differs from its block partner's is
named, not dropped.

**Dry checks, banked before the run.**

- `day39-cpu/reader-dry-day37.log`: the reader over day 37's receipts laid out as the day-39 programs. Contrast b1-b0
  reproduces the `DAY37 VERDICT` lines; the b2 receipts read `inadmissible` (day 37's lines carry no spans term), so
  `DAY39 ADMISSIBLE: 56 of 60 receipts; all=False`.
- `day39-cpu/reader-b2-clause.log`: the b2 clause over A day 30's server log lines: `DAY39 B2-CLAUSE SELFTEST: PASS`.
- `day39-cpu/tick-reader-dry-day37.log`: the tick reader over the same layout reproduces day 38's `DAY38 VERDICT
  q=tick2` lines and `-> consistent with H`.
- `day39-cpu/regime-dry-day37.log`: the regime reader over day 37's hold reproduces day 37's hold and marks lines.

**What the cell cannot say.** One card (the RTX PRO 6000 Blackwell), one model (the 27B artifact), the plain class
under `MEMRA_SERVE_SPEC=0`, the target card's cache and host budgets. Each contrast is the whole `crates/` difference
between its two trees (b1-b0: `091a931c0` to `9717e8d57`, route-contract and latch-close commits included; b2-b1: A
day 30's two commits only); only the DiD and the controls separate the door from the rest, and only as far as the rule
resolves. The tenant's gaps are client-side. Nothing here is compared to the RTX 5090 figures of days 35, 37 and 38,
or to any other day on this card. `executed-not-qualified`; not a qualification.

## 2. The run

- Pre-registration `89c2cd521`, committed at 21:57:26Z and pushed before the first boot (21:58:02Z); the box worktree was detached at that commit
  (`stall/provenance.log`: `tree=89c2cd52112bb1bc9b70118c4cab8b6dff975e85`, `host_utc_offset=+0000`, `model sha256
  MATCH`).
- Builds (section 0), one after the other on the box, `rc=0` each: B0 21:37:08Z to 21:40:16Z, B1 to 21:43:24Z, B2 to
  21:44:10Z (`box/build-B{0,1,2}.log`, `box/builds.log`). The cell's `binary.sha256` matches all three build receipts.
- ONE collector hold on `/tmp/memra-gpu.lock` (`LOCK.json`: owner `collector`, `inherited-flock-same-open-description`),
  no wait and no lock refusal (no `waits.log`). First mark 21:58:02Z, last mark 22:54:04Z: 36 boots, 56.0 minutes, inside
  the 90-minute limit. `stall-day39 rc=0`, the collector's exit 0, `CELL.jsonl` status `executed-not-qualified`.
- 60 receipts, `STALL REPLAY: PASS` on 60 of 60. The compute-apps list before the hold was empty; nothing else ran on
  the card inside it. Lane A's day-31 collector took the lock after the hold ended; it was not touched.
- Receipts: `pro-single-day39/box/` (the builds, the runner and collector logs, `ev/` with per-boot `server.log`,
  `receipt.json`, harness logs, `marks.tsv`, `card.during.csv`, and `collector/command.gpu.csv` at 250 ms).

## 3. The reading (pre-registered readers; every verdict verbatim)

`python3 day39-stall-reading.py pro-single-day39/box/run/stall/ev | tee pro-single-day39/reading.log`, exit 0:

```
DAY39 VERDICT contrast=b1-b0 q=stall: prime o1=-0.1/2.3 o2=+0.0/2.3 under_resolution; demote-off o1=-0.6/2.0 o2=-0.5/1.5 under_resolution; demote-on o1=-31.8/0.8 o2=-31.1/1.2 moved; promote-off o1=-0.3/4.0 o2=-0.3/1.4 under_resolution; promote-on o1=-0.1/0.9 o2=-0.0/0.9 under_resolution; did-demote o1=-31.2/2.1 o2=-30.6/2.0 moved; did-promote o1=+0.3/4.0 o2=+0.3/1.7 under_resolution
DAY39 VERDICT contrast=b2-b1 q=stall: prime o1=+0.0/2.4 o2=-0.0/2.3 under_resolution; demote-off o1=+0.3/1.4 o2=+0.3/1.4 under_resolution; demote-on o1=-41.2/0.7 o2=-41.8/1.2 moved; promote-off o1=-0.0/1.9 o2=+0.2/1.7 under_resolution; promote-on o1=-5.0/0.8 o2=-5.1/0.8 moved; did-demote o1=-41.5/1.6 o2=-42.1/1.9 moved; did-promote o1=-4.9/2.1 o2=-5.3/1.9 moved
DAY39 VERDICT contrast=b1-b0 q=top1_plus_top2: prime o1=-0.1/2.3 o2=-0.0/2.2 under_resolution; demote-off o1=-0.5/1.8 o2=-0.5/1.4 under_resolution; demote-on o1=-75.2/1.7 o2=-73.5/1.5 moved; promote-off o1=-0.3/4.3 o2=-0.3/1.4 under_resolution; promote-on o1=-74.2/0.9 o2=-73.8/2.9 moved; did-demote o1=-74.7/2.5 o2=-73.0/2.1 moved; did-promote o1=-73.9/4.4 o2=-73.5/3.2 moved
DAY39 VERDICT contrast=b2-b1 q=top1_plus_top2: prime o1=+0.1/2.3 o2=+0.0/2.1 under_resolution; demote-off o1=+0.3/1.3 o2=+0.3/1.4 under_resolution; demote-on o1=-41.2/0.7 o2=-41.8/1.2 moved; promote-off o1=-0.0/1.6 o2=+0.2/1.7 under_resolution; promote-on o1=-4.9/0.7 o2=-5.0/2.8 moved; did-demote o1=-41.5/1.5 o2=-42.1/1.8 moved; did-promote o1=-4.9/1.7 o2=-5.2/3.3 moved
DAY39 ADMISSIBLE: 60 of 60 receipts; all=True
```

Per binary, pooled N=40 (`DAY39 BINARY` lines), stall medians (IQR): demote OFF `117.9` (1.3), `117.4` (1.0), `117.7`
(0.9) on b0, b1, b2; demote ON `150.1` (0.3), `118.6` (1.4), `77.1` (0.2); promote OFF `85.3` (3.2), `85.1` (1.3),
`85.2` (1.1); promote ON `82.1` (0.6), `82.0` (0.6), `77.0` (0.5); prime `301.6` on all three. ON minus OFF per tree:
demote `+32.1 unc=1.4 -> isolated` (b0), `+1.2 unc=1.7 -> under_resolution` (b1), `-40.6 unc=1.0 -> isolated` (b2);
promote `-3.3 unc=3.2`, `-3.1 unc=1.4`, `-8.2 unc=1.3`, each `isolated`.

`python3 day39-tick-split.py pro-single-day39/box/run/stall/ev pro-single-day39/reading.log | tee
pro-single-day39/tick-split.log`, exit 0:

```
DAY39 TICK VERDICT contrast=b1-b0 q=tick1: prime o1=-0.0/0.3 o2=-0.0/0.3 under_resolution; demote-off o1=-0.6/2.0 o2=-0.5/1.5 under_resolution; demote-on o1=-0.8/1.7 o2=+0.7/1.6 under_resolution; promote-off o1=-0.3/4.1 o2=-0.3/1.4 under_resolution; promote-on o1=-0.1/0.9 o2=+0.0/0.9 under_resolution; did-demote o1=-0.2/2.6 o2=+1.2/2.2 under_resolution; did-promote o1=+0.2/4.2 o2=+0.3/1.6 under_resolution
DAY39 TICK VERDICT contrast=b2-b1 q=tick1: prime o1=-0.0/0.2 o2=+0.0/0.3 under_resolution; demote-off o1=+0.2/1.4 o2=+0.3/1.5 under_resolution; demote-on o1=-41.2/0.7 o2=-41.8/1.2 moved; promote-off o1=-0.0/1.9 o2=+0.2/1.7 under_resolution; promote-on o1=-4.9/0.8 o2=-5.1/0.8 moved; did-demote o1=-41.5/1.6 o2=-42.1/1.9 moved; did-promote o1=-4.9/2.0 o2=-5.3/1.9 moved
DAY39 TICK VERDICT contrast=b1-b0 q=tick2: prime o1=-0.0/0.2 o2=+0.0/0.2 under_resolution; demote-off o1=+0.0/0.3 o2=-0.1/0.3 under_resolution; demote-on o1=-74.3/0.3 o2=-74.2/0.3 moved; promote-off o1=nd o2=nd not_defined; promote-on o1=nd o2=nd not_defined; did-demote o1=-74.4/0.4 o2=-74.1/0.4 moved; did-promote o1=nd o2=nd not_defined
DAY39 TICK VERDICT contrast=b2-b1 q=tick2: prime o1=+0.0/0.2 o2=-0.0/0.2 under_resolution; demote-off o1=-0.0/0.2 o2=+0.1/0.2 under_resolution; demote-on o1=+0.0/0.2 o2=+0.2/0.3 under_resolution; promote-off o1=nd o2=nd not_defined; promote-on o1=nd o2=nd not_defined; did-demote o1=+0.0/0.3 o2=+0.1/0.4 under_resolution; did-promote o1=nd o2=nd not_defined
DAY39 TICK HYPOTHESIS contrast=b1-b0 P1 demote-on tick2 moved negative: holds; P2 demote-on tick1 under_resolution: holds; P3 did-demote tick2 moved negative: holds; P4 did-demote tick1 under_resolution: holds; P5 demote-off tick1 and tick2 under_resolution: holds (tick1 under_resolution, tick2 under_resolution) -> consistent with H
```

Per tree, demote tick medians (N=40, `DAY39 TICK BINARY`): tick 1 OFF `131.3`, `130.8`, `131.1` and ON `132.0`,
`132.1`, `90.5` (b0, b1, b2); tick 2 OFF `87.7` on all three and ON `163.5`, `89.2`, `89.3`. Promote tick 2 is
`not_defined` because the arm stretches one tick (`40 of 40 runs lack tick 2` for promote OFF on all three trees and
promote ON on b1 and b2; b0's promote ON has tick 2, median `92.8`).

**Described, not ruled on.**

- The ledger (`DAY39 LEDGER`, medians over the ON demote arms' runs, first three demotes N=12, 4th on N=24): b1
  `pre_submit=43.71` / `42.44`, `copy_settle=1.21` / `1.28`, `take_back_publish=1.08` / `1.08`,
  `owner_in_completion=44.81` / `43.53`, `owner_held=46.03` / `44.80`, `hashed_in=73.3` / `73.2`; b2
  `pre_submit=1.17` / `1.15`, `copy_settle=1.29` / `1.36`, `take_back_publish=1.08` / `1.08`,
  `owner_in_completion=2.25` / `2.25`, `owner_held=3.55` / `3.59`, `hashed_in=107.3` / `104.8`,
  `landed_polls_median=3` (b1 1). `parked_hits_sum=0` and `reparks_sum=0` on both; `payloads=[98]` on both. b0 prints
  no ledger line (0 in all four ON boots; `demote_published_lines=21` in each).
- The b2 spans (`pro-single-day39/spans-census.log`): every one of the 84 copy-complete lines of the four b2 ON boots
  carries `items=128 (32 KV, 96 f32 spans)`, and each boot has 21 D2H receipt lines ending `96 f32 spans landed under
  the ticket and taken back before the retire`; b1's 84 copy-complete lines carry no spans term and no such tail.
- Sizes side by side, not timed against a gap: b2-b1's demote-on tick-1 move (`-41.2` / `-41.8`) and b1's pre-submit
  minus b2's (42.44 minus 1.15 = 41.29 on the 4th on); b1's remaining tick-2 ON minus OFF (`+1.6 unc=0.3`) and its
  `copy_settle` 1.28. Within this cell b1's pre-submit shows no first-touch step: `43.71` on the first three demotes
  and `42.44` from the 4th on. The packet's section 4 ledger row records such a step in the day-28 double-park cell's
  ON boots (a different cell shape, and this record does not assert the same box); the two are named, not compared as
  timings, and what separates them is owed (section 7).
- Attribution (`DAY39 ATTRIBUTION`, ON demote, N=9 per pass): `demote_in` medians 132.4 to 134.0 (b0), 148.8 to 149.6
  (b1), 135.8 to 136.2 (b2); completion 57.5 to 58.8 (b0), 57.8 to 58.4 (b1), 16.8 to 16.9 (b2); OFF `demote_in` 41.6
  to 42.7 on all three.

**The regime** (`python3 day39-regime.py pro-single-day39/box/run/stall/collector pro-single-day39/box/run/stall/ev
+0000 | tee pro-single-day39/regime.log`, 13,410 samples at 250 ms over the hold):

```
DAY39 REGIME hold: samples=13410 temp_c=33..59 (median 50) power_w=33.50..499.83 (median 317.76) clocks_sm_mhz=180..2422 (median 2422) mem_used_mib=0..19573 power_limit=['600.00 W']
```

Per program: 2227 to 2245 samples, temperature median 50 C in all six (range 33..59 in p1-b0, the hold's cold start;
40 or 41..59 in the other five), power median 317.67 to 317.98 W, SM clock median 2422 MHz in all six (minimum 180 in
p1-b0 at idle before the first boot, 2340 to 2355 elsewhere). No program's regime differs from its block partner's on
these figures.

**One reader fix after the run, a parse, not a rule.** `day39-regime.py` failed on its first run over the box's
`marks.tsv`: `ValueError: unconverted data remains: Z`. The box's `date` writes 3 fraction digits where the local
rig's writes 9, and the day-37 parser cut at a fixed 26 characters. The fix pads or cuts the fraction to 6 digits
(`day39-cpu/diffs/regime-postrun-parse-fix.diff`); the dry check over day 37's hold still reproduces day 37's hold and
marks lines. The regime reader decides nothing; no rule, quantity, reader or admissibility clause of the stall or tick
readers changed after the run.

## 4. What the cell cannot say

One card, one model (the 27B), the plain 64-token class under `MEMRA_SERVE_SPEC=0`, this card's cache and host
budgets. Each contrast is the whole `crates/` difference between its two trees (b1-b0 includes route-contract and
latch-close commits; b2-b1 is A day 30's two commits only); the DiD and the controls separate the door from the rest
only as far as the rule resolves, and here every control (prime, demote OFF, promote OFF) is `under_resolution` in
both contrasts on both quantities. b2-b1 carried no prediction; its lines are measurements under the rule, not a
tested hypothesis. The tenant's gaps are client-side; the ledger and attribution segments are the server's lines and
are not timed against a gap. No figure here is compared to the RTX 5090 cells of days 35, 37 and 38 or to earlier days
on this card. `executed-not-qualified`; not a qualification.

## 5. Commits

- `a20e4990d` merge of integ47; `d036db237` the packet status fix; `04584a2a5` section 0 and the build script;
  `89c2cd521` the pre-registration (pushed before the first boot); the receipts and records commit follows.

## 6. Checks

`bash tools/check-flags.sh`, `bash tools/check-conflict-markers.sh`, `git diff --check`, `python3
tools/check-public-boundary.py check`, zero em dashes in added lines: run before each push (section 5), and all pass
on the records commit's staged tree.

## 7. Owed and open

- The target card's promote class on tick 2: `not_defined` on b1 and b2 (the arm stretches one tick).
- Why b1's pre-submit shows no first-touch step in this cell (42 to 44 ms on every demote, the 4th on included) where
  the day-28 double-park cell's ledger row shows one: not separated (different cell shapes, no same-box claim; a
  per-demote allocation line would name it, and that is engine code, lane A's).
- The b2 `hashed_in` of 104.8 to 107.3 ms against b1's 73.2 to 73.3 (the helper's time per 157.9 MB, off the tick):
  described, not attributed.
- Unchanged: the H2D and D2D halves of Move 2 owed item 1, the governor charge of the 157.9 MB staging, the
  strong-form receipt, the span-refusal fault cell and returning the staging after a refused receipt (`A/DAY30.md`
  section 9); the owner's decisions at 2026-09-23, 2026-10-04, 2026-10-05 (contracts door) and 2026-10-06; the
  double-park slice question; the 9B entry's conv, ssm and hidden split (engine line); an always-admitted prime arm
  on the RTX 5090 class.

## 8. Cleanup and budget

- The box: before removal, a SHA-256 manifest of the box's run directory and build logs (339 files) matched the banked
  copy under `pro-single-day39/box/` file for file. Then this lane's box scratch was removed whole: the transfer
  bundle, the run directory, the build logs and the three binary copies (their SHA-256s stay in `box/builds.log`;
  each is rebuildable from its SHA by `day39-box-build.sh`); the four temporary refs in the lane's box worktree were
  deleted (0 left). The lane's box worktree stays, detached at `89c2cd521`, clean. No server, collector or lock of
  this lane is left on the box. Lane A's day-31 server on the card was not inspected beyond `nvidia-smi`'s listing,
  and not touched.
- Local: the scratch directory, the bundle, the check script and its logs under `/tmp`, and the four temporary refs
  are removed at close. No worktree was created.
- Budget: started 21:24Z, the records commit about 23:10Z; about 1.8 agent-hours of the 4.
