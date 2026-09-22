# Session C day 37: the demote-class tenant-stall cell on the option (a) code, RTX 5090 class, two binaries in one hold

Lane `lane/spill-c-20260919`, checkout `wt-spill-c`, start `558e4d022` = remote, no tracked edit. Every push today in
the announced `MEMRA_RELEASE_QUALIFICATION_MODE=development` mode (the hook prints `UNQUALIFIED DEVELOPMENT:
refs/heads/lane/spill-c-20260919 at <sha>; no GPU qualification claimed` and records the skip in the clone's
`.git/memra-gate-skips.log`). Every cell below is `executed-not-qualified` development evidence. No commit on main, no
PR, no engine change, no `docs/` registry edit, no recommendation. Today's card is the LOCAL RTX 5090 Laptop GPU with
the 9B artifact (`Qwen3.5-9B-NVFP4-MTP-GGUF.gguf`), lock `/tmp/memra-5090.lock`; every number here is this card's own
and is compared to nothing measured on the target card. The target card is lane A's today; not touched.

## 0. Merges and builds (first action)

- `d7bf406af`: `origin/main` at `189c91b15` (#595) merged. `8b889dcdf`: `origin/lane/spill-integ45-20260922` at
  `298f7d052` (integ45: A days 28 and 29, `c26255bc7` the bounded helper close at the latch) merged. Both clean;
  `check-conflict-markers: OK`. Pushed `558e4d022..8b889dcdf`. **`8b889dcdf` is the option (a) tree.** Its `crates/`
  differs from `091a931c0` (day 35's tree) by six engine commits, no merge among them (`git log --oneline --no-merges
  091a931c0..8b889dcdf -- crates/`): the door's `45f824a75` (A day 28, option (a)), `867655368` (A day 29, 2a),
  `c26255bc7` (integ45), and the route-contract commits `66669acc6`, `8b1debbe0`, `3cf270bca` (memra#504); 4 files,
  `2356 insertions(+), 47 deletions(-)`.
- **Base binary.** `091a931c0` in a detached worktree `wt-c37-base` (removed at close), `cargo build --release -p
  memra-server` under `systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=20G`, `rc=0 wall_s=178`, SHA-256
  `3278b2401b1ca4bae81e9ce3d82b73682150bb8bf93c2264fd32f6b0655f666c` (`day37-cpu/build-base.log`). Day 35's binary of
  the same tree printed `7ce7bf78...` from this worktree's `target/`; the two are different files of one tree, and
  every boot's `[server] build:` line names the git id it ran.
- **Option (a) binary.** `8b889dcdf` in `wt-spill-c`, same scope, `rc=0 wall_s=171`, SHA-256
  `f7ad8c607b047a3f7fd17ee6b0309156e682a4ecf89a8760f02db53ccc3da705` (`day37-cpu/build-opta.log`). `free -g` read 51 GB
  available at the start of each build.

## 1. Pre-registration (committed and pushed before the first boot)

**What is owed.** `DAY36.md` section 4: "No demote-class tenant-stall cell has run on the day-28 and day-29 code on
either card; every day-28 and day-29 stall figure is the promote-then-hit shape." The packet's item 2 scope line names
the same gap. Today closes it on the RTX 5090 class only.

**Unchanged from day 35.** The harness `day35_stall_cell.py` byte-for-byte (SHA-256
`9c9b38782db8798d16fc9189bdeda9c4fe4742251584abedd55a261615ae0ccc`, the figure day 35's `CELL.txt` carries). The five
arms (prime, demote OFF, demote ON, promote OFF, promote ON), the shapes (tenant fired at its 24th token, the prime,
demote and promote intruders), N=5 per arm per order with both orders inside every harness run, the boot env
(`MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 MEMRA_SERVE_SPEC=0 MEMRA_COMPAT=openai`, cache 64 MB, host 8192 MB, the prime
boot `MEMRA_PREFIX_CACHE_MB=0 MEMRA_KV_HOST_MB=0`, the ON boots `MEMRA_KV_HOST_CONTRACTS=1`), port 18131, the tenant's
`max_tokens` 400, the per-run landing rule (`fired_at_ms + wall_ms <= tenant_wall_ms`), the BAD-line list, the
promote shape (1,1,1,1,0), the bounded card-free wait and the lock-refusal retry (15 waits of 120 s).

**The one change: two binaries in ONE lock hold, interleaved in both orders.** Four programs in one collector hold on
`/tmp/memra-5090.lock` (`tools/tier-battery.py --rig rtx5090 --external-lock`): **p1-base, p2-opta, p3-opta,
p4-base**. Each program is day 35's six boots in day 35's order (pass 1 prime, off, on; pass 2 on, off, prime). The
scripts are day 35's copied to day-37 names where a path or label had to change: `day37-stall-cell.sh` (the program
loop, the two binaries, `ev/<program>/pass{1,2}/<kind>/`, the `CELL.txt` labels and mark names),
`day37-local-run.sh` (two binaries; the collector timeout 5400 s against day 35's 3600 s), `day37-stall-reading.py`
(below). The diffs are banked as `rtx5090-day37/diffs/{cell,runner,reader}.diff`.

**Wall time.** Day 35's program took 655.4 s first mark to last mark (`day37-cpu/wall-estimate.log`, over
`rtx5090-day35/stall/ev/marks.tsv`); four programs estimate 2622 s = 43.7 min, under the 90-minute limit, so both
orders run and nothing is cut. The collector's 5400 s timeout ends the hold at 90 minutes if it runs long; programs not
finished by then are reported as missing and their arms `inadmissible`.

**Admissibility per receipt.** Day 35's, unchanged, except the ON demote line set, which is per tree because the
option (a) tree does not print `demote published off the tick` for an image with heap payloads (`worker.rs` on
`8b889dcdf`, the `payloads.is_empty()` branch):

- base: every ON demote run with a demote line has `demote submitted off the tick`, a `D2H receipt ... require=ok` and
  `demote published off the tick` (day 35's three lines).
- option (a): every ON demote run with a demote line has `demote submitted off the tick`, a `D2H receipt ...
  require=ok`, `demote copy complete off the tick` and `demote digests landed off the tick`, and zero `demote
  published off the tick`.

**The rule (what counts as moved), written from day 35's reading.** For each arm (prime, demote-off, demote-on,
promote-off, promote-on) and each order block (o1 = p2-opta against p1-base, o2 = p3-opta against p4-base):
d = stall_median(option (a) program) minus stall_median(base program), each side the program's two passes pooled
(N=20), unc = the quadrature of the two IQRs, `isolated` when |d| > unc (day 35's per-pass rule, applied across
binaries). An arm has **moved** when both blocks are isolated with the same sign; `order_split` when both are
isolated with opposite signs; `under_resolution` otherwise; `inadmissible` when any receipt of either block is. The
door-attributable reading is the difference in differences per class per block, (ON minus OFF)_opta minus (ON minus
OFF)_base, unc the quadrature of the four IQRs, with the same four outcomes. The primary lines are the demote-on arm
and the demote DiD. The prime and OFF arms are controls for the part of the tree difference that is not the door: a
demote-on move with the demote-off control moved in the same blocks is read through the DiD, not as the door's.

**What is read (`day37-stall-reading.py`, fixed now).**

1. Per program per pass: day 35's lines (`stall_median`, IQR, `on_minus_off`, isolated or under_resolution), and the
   `STALL REPLAY` verdict per receipt.
2. Per binary per arm, pooled over its four receipts (N=40): `stall_median`, IQR, N, ON minus OFF per class with unc,
   classified only when all eight receipts of the pair are admissible.
3. Option (a) minus base per arm per block, and the verdict per arm, under the rule above.
4. The DiD per class per block under the rule above.
5. The secondary quantity, from day 35's post-hoc (the door's demote share lands on the SECOND stretched tick):
   `top1_plus_top2`, the tenant's two largest ITL gaps per run summed, readings 2 to 4 on it with the same rule.
6. The option (a) ledger: every `demote digests landed off the tick` line in the option (a) ON demote arms' run lines,
   split into the boot's first three demotes and the 4th onward (day 36's form): medians of `pre-submit`, `copy
   settle`, `hashing polls`, `take-back bind and publish`, owner `in-completion`, owner held, `hashed in`, wall; the
   parked-hit and re-park sums; per ON boot the counts of ledger lines, `demote published off the tick`, `hit parked on
   a Hashing entry` and `hash helper detached`. The base tree's ledger count is printed (expected 0).
7. Day 35's attribution per tree: `demote_in` against the completion figure (base `demote published off the tick`,
   option (a) `demote copy complete off the tick`), described, not ruled on.

Dry checks, both banked: over day 35's receipts laid out as p1-base and p4-base (`day37-cpu/reader-dry-day35.log`),
the per-pass lines reproduce day 35's verdict line (demote pass 1 `on_minus_off=+3.3 unc=3.3`, pass 2 `+0.3` and
`2.7`; promote `+1.6`/`12.5`, `+2.6`/`2.7`; prime pass 1 `inadmissible`); the ledger regex parses both `demote digests
landed` lines of A's day-29 identity host-on server log with `nomatch=0` (`day37-cpu/reader-ledger-regex.log`).

**What the cell cannot say.** One card class (the RTX 5090 Laptop GPU), one model (the 9B NVFP4 artifact), the plain
class (the 64-token entry, 54.6 MB, 16 items) under `MEMRA_SERVE_SPEC=0`. Not the spec draft-bearing entry class, and
not the target card's demote class on the option (a) tree. Option (a) minus base is the whole `crates/` difference
between `091a931c0` and `8b889dcdf`, route-contract commits included; only the DiD and the controls separate the door
from the rest, and they separate it only as far as the rule resolves. The tenant's gaps are client-side; the
owner-thread segments come from the option (a) ledger lines, which the base tree does not print. No figure here is
compared across days or cards; day 35's receipts appear only as the reader's dry check. Not a qualification.

## 2. The run

`day37-local-run.sh` found the card free on its first look (no `waits.log`), took the lock once and ran the four
programs in one collector hold, 19:47:19Z to 20:31:34Z (44.3 min against the 43.7 estimate), `stall-day37 rc=0`,
`runner rc=0` (`runner.log`, `stall/ev/exit.txt`, `stall/progress.log`). Receipts under `rtx5090-day37/stall/ev/`,
the collector's under `stall/collector/`.

- **Binaries.** Every boot's `[server] build:` line names its tree: 6 of 6 boots of p1-base and p4-base print `git:
  091a931c023a`, 6 of 6 of p2-opta and p3-opta print `git: 8b889dcdf53e` (`provenance.log`). `CELL.txt`'s
  `opta_tree=a50922b27` is the worktree's HEAD when the cell ran, the pre-registration commit; the option (a) binary was
  built at `8b889dcdf`, and `git diff --stat 8b889dcdf a50922b27 -- crates/` prints 0 lines (`provenance.log`).
  `binary.sha256` carries the two SHA-256 figures of section 0.
- **Replays and admissibility.** 40 `STALL REPLAY: PASS` in `replays.log`, 0 lines matching `fail`
  (`provenance.log`); `DAY37 ADMISSIBLE: 40 of 40 receipts; all=True` (`reading.log`). Every ON demote run carries its
  tree's line set: per ON boot the base tree prints 21 `demote published off the tick` lines and 0 ledger lines, the
  option (a) tree 21 `demote digests landed off the tick` lines and 0 `demote published off the tick`; 0 `hit parked on
  a Hashing entry` and 0 `hash helper detached` in all eight ON boots (the 8 `DAY37 LEDGER-BOOT` lines).
- **Regime** (`regime.log`, the collector's 250 ms CSV). The hold: 10522 samples, 51 to 89 C (median 86), 8.74 to
  174.32 W (median 165.49), card-wide memory 15 to 9753 MiB, `power.limit [N/A]`. Per program: p1-base 51 to 89 C
  (2611 samples, power median 171.37 W), p2-opta 76 to 89 C (2614, 168.18 W), p3-opta 76 to 89 C (2634, 162.71 W),
  p4-base 76 to 89 C (2654, 159.51 W); temperature median 86 C in each. The 51 C minimum is the hold's first boot, p1-base
  pass 1 prime; every later boot started warm. The card-wide memory never passed one 9B server's 9753 MiB, and the cell's
  `compute-apps.before.csv` and `compute-apps.after.csv`, written inside the hold, list no app: no co-tenant this day.
  After the lock was released, nvidia-smi's listing showed one compute app that this lane did not start; it was not
  inspected or signalled.

## 3. The reading

Every figure in this section is a line of `rtx5090-day37/reading.log` (command on its first line: `python3
research/spill-c-20260919/day37-stall-reading.py research/spill-c-20260919/rtx5090-day37/stall/ev`), unless a
different log is named.

**Per binary per arm**, each pooled over its four receipts (N=40; `DAY37 BINARY` lines), median (IQR):

| Arm | base `stall` | option (a) `stall` | base `top1_plus_top2` | option (a) `top1_plus_top2` |
|---|---|---|---|---|
| prime | 280.5 (33.9) | 279.5 (13.3) | 556.8 (56.7) | 558.2 (24.9) |
| demote-off | 61.6 (3.2) | 63.8 (4.6) | 110.7 (5.2) | 114.5 (8.3) |
| demote-on | 66.0 (3.7) | 63.9 (3.3) | 143.7 (6.6) | 121.4 (5.0) |
| promote-off | 49.6 (5.8) | 48.3 (2.4) | 67.6 (7.0) | 65.7 (2.7) |
| promote-on | 52.1 (5.0) | 50.9 (1.7) | 99.4 (8.4) | 76.9 (3.5) |

**ON minus OFF per binary** (N=40 each side):

| Class | base `stall` | option (a) `stall` | base `top1_plus_top2` | option (a) `top1_plus_top2` |
|---|---|---|---|---|
| demote | `+4.4 unc=4.9 -> under_resolution` | `+0.1 unc=5.7 -> under_resolution` | `+33.0 unc=8.4 -> isolated` | `+6.9 unc=9.7 -> under_resolution` |
| promote | `+2.5 unc=7.6 -> under_resolution` | `+2.6 unc=2.9 -> under_resolution` | `+31.8 unc=10.9 -> isolated` | `+11.1 unc=4.4 -> isolated` |

**Option (a) minus base per arm per block, and the DiD** (N=20 per program; `d` / `unc`; o1 = p2-opta against
p1-base, o2 = p3-opta against p4-base):

| Arm | `stall` o1 | `stall` o2 | `top1_plus_top2` o1 | `top1_plus_top2` o2 |
|---|---|---|---|---|
| prime | +27.9 / 51.4 | -6.4 / 15.0 | +45.6 / 97.7 | -11.6 / 24.4 |
| demote-off | +1.3 / 4.1 | +2.6 / 5.5 | +2.5 / 6.4 | +4.5 / 9.4 |
| demote-on | -1.0 / 4.2 | -3.8 / 4.7 | -21.7 / 5.2 isolated | -24.5 / 8.5 isolated |
| promote-off | +0.5 / 7.6 | -1.9 / 3.4 | +0.9 / 11.3 | -2.4 / 3.3 |
| promote-on | -1.4 / 4.4 | -1.3 / 7.4 | -23.1 / 8.7 isolated | -21.4 / 9.2 isolated |
| DiD demote | -2.3 / 5.8 | -6.4 / 7.3 | -24.2 / 8.2 isolated | -29.0 / 12.7 isolated |
| DiD promote | -1.8 / 8.8 | +0.7 / 8.1 | -24.0 / 14.2 isolated | -19.1 / 9.8 isolated |

Cells not marked `isolated` read `under_resolution`. The DiD terms per block, `(on-off)_opta` against
`(on-off)_base`: `stall` demote `+1.6` / `+4.0` (o1) and `-1.4` / `+5.0` (o2); `top1_plus_top2` demote `+9.6` / `+33.9`
and `+4.2` / `+33.3`, promote `+10.9` / `+34.9` and `+11.8` / `+30.9`.

**The verdicts, verbatim:**

```
DAY37 VERDICT q=stall: prime o1=+27.9/51.4 o2=-6.4/15.0 under_resolution; demote-off o1=+1.3/4.1 o2=+2.6/5.5 under_resolution; demote-on o1=-1.0/4.2 o2=-3.8/4.7 under_resolution; promote-off o1=+0.5/7.6 o2=-1.9/3.4 under_resolution; promote-on o1=-1.4/4.4 o2=-1.3/7.4 under_resolution; did-demote o1=-2.3/5.8 o2=-6.4/7.3 under_resolution; did-promote o1=-1.8/8.8 o2=+0.7/8.1 under_resolution
DAY37 VERDICT q=top1_plus_top2: prime o1=+45.6/97.7 o2=-11.6/24.4 under_resolution; demote-off o1=+2.5/6.4 o2=+4.5/9.4 under_resolution; demote-on o1=-21.7/5.2 o2=-24.5/8.5 moved; promote-off o1=+0.9/11.3 o2=-2.4/3.3 under_resolution; promote-on o1=-23.1/8.7 o2=-21.4/9.2 moved; did-demote o1=-24.2/8.2 o2=-29.0/12.7 moved; did-promote o1=-24.0/14.2 o2=-19.1/9.8 moved
```

**Day 35's per-pass lines.** 16 `DAY37 STALL ... on_minus_off` lines (two passes, two classes, four programs), 4
`isolated` (`grep '^DAY37 STALL.*on_minus_off' reading.log | grep -c isolated`): base p1-base pass 1 demote `+4.5
unc=2.5` and promote `+5.7 unc=2.9`, p4-base pass 1 demote `+7.2 unc=4.1`; option (a) p3-opta pass 2 promote `+2.6
unc=2.1`. The other 12 read `under_resolution`. The prime passes read 229.0 (p1-base pass 1, the hold's first boot)
and 269.7 to 294.2 in the other seven; the base prime arm's IQR of 33.9 over N=40 comes from that pass. The cause of
the low first pass is not separated here.

**The option (a) ledger** (the demote arm's run lines of the four option (a) ON boots; the base tree prints no
ledger), verbatim:

```
DAY37 LEDGER part=first3 N=12 pre_submit=23.49 copy_settle=8.48 hashing_polls=0.00 take_back_publish=8.36 owner_in_completion=31.88 owner_held=40.29 hashed_in=12.9 wall=91.7 landed_polls_median=1 settle_polls_median=1 parked_hits_sum=0 reparks_sum=0 payloads=[50] modes=['tick-top poll'] (medians over the ON demote arms' run lines)
DAY37 LEDGER part=4th_on N=24 pre_submit=25.06 copy_settle=8.30 hashing_polls=0.00 take_back_publish=8.28 owner_in_completion=33.48 owner_held=41.85 hashed_in=12.9 wall=95.3 landed_polls_median=1 settle_polls_median=1 parked_hits_sum=0 reparks_sum=0 payloads=[50] modes=['tick-top poll'] (medians over the ON demote arms' run lines)
```

`first3` is a boot's first three ledger lines by ticket order, `4th_on` the rest of the demote arm's (3 and 6 per
boot). On this card the pre-submit segment reads about the same on the first three demotes of a boot as on the later
ones (23.49 against 25.06); the helper's `hashed in` is 12.9 ms per `50 payloads (53.7MB)` in both parts, off the tick,
and every ticket landed after 1 poll with 0 parked hits.

**Day 35's attribution per tree** (`DAY37 ATTRIBUTION`, N=9 per pass, described, not ruled on). The base tree's ON
demote: `demote_in` 63.5 to 65.9, completion (`demote published off the tick`) 41.6 to 43.6, `in_minus_completion`
21.8 to 22.8 over its four passes. The option (a) tree's: `demote_in` 93.3 to 95.7, completion (`demote copy complete
off the tick`) 40.3 to 42.3, `in_minus_completion` 52.2 to 54.9. On the option (a) tree `demote_in` spans t0 to
publication (the server's `demote: ... in Y ms` equals the ledger's `wall` for the same ticket: seq=2 of p2-opta pass
1 prints `in 86.1ms` and `wall 86.1ms t0 to publication`), the helper's hash and its landing poll inside it, so its
`in_minus_completion` is not the owner-thread share the base figure is; the owner-thread share on that tree is the
ledger's segments. OFF `demote_in` 23.3 to 26.5 on both trees.

**Read under the pre-registered rule.** The primary lines, the demote-on arm and the demote DiD by the worst-tick
rule, are `under_resolution` in both blocks (`-1.0 / 4.2`, `-3.8 / 4.7`; `-2.3 / 5.8`, `-6.4 / 7.3`). By the
secondary quantity the demote-on arm `moved` (`-21.7 / 5.2`, `-24.5 / 8.5`), its OFF control did not (`+2.5 / 6.4`,
`+4.5 / 9.4`), and the demote DiD `moved` (`-24.2 / 8.2`, `-29.0 / 12.7`); the promote-on arm and the promote DiD
`moved` the same way (`-23.1 / 8.7`, `-21.4 / 9.2`; `-24.0 / 14.2`, `-19.1 / 9.8`), and the prime and both OFF controls
read `under_resolution`. Within each binary, the door's ON minus OFF on the secondary quantity reads demote `+33.0
isolated` on the base tree and `+6.9 under_resolution` on the option (a) tree, promote `+31.8 isolated` and `+11.1
isolated`. The difference is the whole `crates/` difference between the two trees; the DiD and the controls are what
the rule offers to separate the door from the route-contract commits, and they resolve as stated above.

**What this cell does not say.** The RTX 5090 Laptop GPU, the 9B NVFP4 artifact, the plain 64-token class under
`MEMRA_SERVE_SPEC=0` only. Not the spec draft-bearing class, and not the target card's demote class on the option (a)
tree, which no cell has measured. The reader does not split the two gaps it sums, so which of the tenant's two stretched
ticks moved is not read here. No figure is compared with day 35's receipts or with the target card. Not a
qualification.

## 4. The counting rule, applied

| Figure | Command | Output |
|---|---|---|
| Every stall, `top1_plus_top2`, ON minus OFF, option (a) minus base, DiD, verdict, per-pass, ledger, attribution and admissibility figure of section 3 | `python3 research/spill-c-20260919/day37-stall-reading.py research/spill-c-20260919/rtx5090-day37/stall/ev` | `rtx5090-day37/reading.log` |
| The regime, whole hold and per program | `python3 research/spill-c-20260919/day37-regime.py research/spill-c-20260919/rtx5090-day37/stall/collector research/spill-c-20260919/rtx5090-day37/stall/ev` | `rtx5090-day37/regime.log` |
| The build ids per program, the 0-line `crates/` diff, 40 replays PASS and 0 `fail` lines | the three commands on the `# command:` lines | `rtx5090-day37/provenance.log` |
| 4 of 16 per-pass lines `isolated` | `grep '^DAY37 STALL.*on_minus_off' reading.log \| grep -c isolated` (and `grep -c .` for 16) | stated here |
| The `in 86.1ms` and `wall 86.1ms` pair | `grep -m6 'prefix-host\] demote' rtx5090-day37/stall/ev/p2-opta/pass1/on/server.log` | the server log |
| The hold's start, end and rc | `runner.log`, `stall/ev/exit.txt`, `stall/progress.log`, `stall/ev/marks.tsv` | as named |
| Option (a)'s code (`45f824a75`, `867655368`) an ancestor of `origin/main` `0c86309bd` | the `# command:` line | `rtx5090-day37/ancestry.log` |

`day37-regime.py` is new today and read the hold's own CSV only; it was written after the run and is labelled as such
here (the regime is a condition, not a reading of the rule).

## 5. Commits

- `d7bf406af`: merge of `origin/main` `189c91b15` (#595) into the lane.
- `8b889dcdf`: merge of `origin/lane/spill-integ45-20260922` `298f7d052` (`c26255bc7`, the bounded helper join at the latch); the option (a) tree the opta binary was built from.
- `a50922b27`: the pre-registration (section 1), the cell, runner and reader with their diffs against day 35, both builds' logs; pushed before the first boot.
- `adc142c4d`: the run's receipts, server logs, driver log, `reading.log`, `regime.log`, `provenance.log` and `day37-regime.py`.
- The records commit after `adc142c4d`: this file's sections 3 to 8, `DOOR-DECISION-PACKET.md`, `HOSTPREFIX-DOOR.md` section E, `STATE.md`, the `research/INDEX.md` row and `rtx5090-day37/ancestry.log`.

Every push ran as `MEMRA_RELEASE_QUALIFICATION_MODE=development git push origin lane/spill-c-20260919` (announced; the hook logs the mode). No commit on `main`, no PR, no engine change, no `docs/` registry edit.

## 6. Checks

Each ran in its own `if ! ...; then exit; fi` line on the records tree before its commit:

- `bash tools/check-flags.sh`: "check-flags: no uncovered runtime names" and "every runtime MEMRA_* name resolves against 'docs/FLAGS.md' (no grandfather list)".
- `bash tools/check-conflict-markers.sh`: "check-conflict-markers: OK (no conflict marker line in tracked source or docs)".
- `git diff --check`: clean.
- Em dashes (U+2014) in added lines, `git diff 8b889dcdf -- research/spill-c-20260919/ research/INDEX.md | grep '^+' | grep -cP '\x{2014}'`: 24, one per raw `server.log` (24 logs), each in the engine's own `[gpu-watch] Xid source:` startup line, which prints U+2014 between `(/dev/kmsg unreadable` and `kernel.dmesg_restrict)`. Raw receipts are banked as written and the engine string is outside this lane's scope. The same count with the 24 raw server logs excluded (`':(exclude)research/spill-c-20260919/rtx5090-day37/stall/ev/*/*/*/server.log'`): 0.
- `python3 tools/check-public-boundary.py check`: "public-boundary: 604 matches (604 grandfathered, 0 new)."

## 7. Owed and open

- The target card's demote-class tenant-stall cell on the option (a) tree: no cell has run (the packet's item 2 scope, item 7 and section 6 say so). The target card is lane A's today; this lane did not touch it.
- A per-tick split of the two summed gaps: the reader sums them and does not name which tick moved (section 3). A reader change with its own pre-registration; not run.
- The prime control's low first pass (229.0 at 51 C on the hold's first boot against 269.7 to 294.2 elsewhere): cause not separated; an always-admitted prime arm on this card class is still a new pre-registration, not run.
- Unchanged from day 36: the owner's decisions at 2026-09-23, 2026-10-04, 2026-10-05 and 2026-10-06; Move 2 owed item 1 (A day 30, running in A's lane); the 9B entry's KV byte split; the double-park slice question.

## 8. Cleanup and budget

- `git worktree remove /home/avifenesh/projects/wt-c37-base` (the detached base checkout; no branch to delete); `/tmp/c37-*` removed.
- At close: no server, lock or GPU process of this lane's. A foreign compute app on the card after the lock release (another lane's server) was not inspected or signalled.
- Agent-hours: from the first merge (`d7bf406af`, 19:29Z) to the records push, about 1.5 of the 3-hour budget.
