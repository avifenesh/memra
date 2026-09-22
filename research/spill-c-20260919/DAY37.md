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
