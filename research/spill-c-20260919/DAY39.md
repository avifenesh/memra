# WP-C day 39 (2026-09-23): the target card's demote-class tenant-stall cell

Lane `lane/spill-c-20260919`. Start tip `18c900fc7`; `origin/lane/spill-integ47-20260923` (`160929a92`, A day 30
and C day 38) merged as `a20e4990d`; the packet status fix is `d036db237` (A day 30 landed on integ47, not on
`origin/main` `9717e8d57`; `day39-cpu/ancestry.log`). Every cell here is `executed-not-qualified` development
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
