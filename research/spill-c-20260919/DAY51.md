# WP-C day 51 (2026-09-24): the MoE slot cache door's deciding cell, pre-registered before any rung is measured

`OWED.md` C1 step (c). The door's decide-by is 2026-10-04 (`MOE-SLOT-CACHE-DOOR.md`). Days 43 to 50 landed ten
improvements, each with its own pre-registered cell (`resid`, `mapped`, `fill`, `nodrain`, `pinned`, `small`,
`install`, `prefetch`), all queued on the RTX 5090 and none run at the time of writing, so nothing below is shaped by
their results. The deciding cell runs on the tuned door: the tree after every improvement's verdict (a rung that fails
its own clauses is fixed or reverted first, and the binary is named in section 2 before this cell runs). Tree at
start: `9602a84d5`.

## 1. Pre-registration

**What is decided.** Door hygiene (`CLAUDE.md`, "Door hygiene"): a default-OFF door whose receipt is negative, flat,
neutral or no-go is deleted in the same lane; a winner goes to the owner as a promotion call. This cell's verdict,
per card, is one of `door_wins`, `door_flat` or `door_loses` against the naked-default legacy program.

**Correctness first, per card, on the final binary (pass/fail; a failure voids the timing verdict).**

- G1, the hash lock still refuses: day 18's `hashlock` cell unchanged on a non-approved single-shard artifact (the
  Qwen3.5-9B NVFP4 MTP GGUF on the RTX 5090; the 27B on the target card): the door run exits 1 with the last line
  `Error: "experts-via-tier artifact SHA256 mismatch"` and no bracketed door line; the control run `MATCH`.
- G2, speculative self-consistency under the door: `run-spec` on the approved artifact with its MTP head,
  `--experts-via-tier --expert-bank-host-bytes=17179869184`, K=1..8, the day-11 prompt set: `=== SELF-CONSISTENCY
  PASS ===` and every K `self-consistency: PASS (identical to plain target)` (the one-numeric-program rule across
  spec-verify and plain, now with event-retired leases, the fill and the prefetch).
- G3, the integrity terms inside the timing cell: every door run's tape equals every legacy run's; `MATCH`; the
  installer's `catalog_sha256`, `records` and `records_sha256` equal day 40's; `fill_refused=0`; the trace
  consistency (`physical_reads` equals the `hit=false` lines, trace lines equal host demands).

**The timing cell `decide`.** Day 18's `overlap` environment (`MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32
MEMRA_MOE_SLOTS=9986`, prompt `55 88 13`, the approved artifact), one binary (the final tree) for every arm, three
arms: OFF (the legacy naked default: no door, no prefetch); ON (`--experts-via-tier
--expert-bank-host-bytes=17179869184`, no stage clock, so the door runs without its diagnostic's cost); REF (the
legacy with `MEMRA_MOE_PREFETCH=1`, the experimental legacy prefetch, a reference arm the verdict does not read,
reported because the door's I4 has this legacy counterpart). Order 1 (OFF, ON, REF) x 5, order 2 (REF, ON, OFF) x 5,
N=5 per arm per order, N=10 pooled, one collector hold, 250 ms telemetry, raw logs and binary SHA-256 banked, the
runner under the 1200% CPU cap.

**Readings.** Per arm: gen-only decode seconds (the `generated 32 tokens in Xs` line, day 18's observation), the
steady window seconds (day 40's line), `install_s` (ON), the process wall. `noise` for a pair is the larger of the
two arms' IQRs over their 10 runs.

**The rule (gen-only decode, the day-18 observation; the window read the same way beside it).**
- `door_wins` iff `median(ON) < median(OFF) - noise` in order 1, in order 2 and pooled.
- `door_loses` iff `median(ON) > median(OFF) + noise` in both orders.
- `door_flat` otherwise.
Reported beside the verdict, no clause: `decode_ratio = median(ON) / median(OFF)` pooled and per order (day 18's
figure, 5.743 on the target card), the window's verdict by the same rule, ON's `install_s` against OFF's load, REF
against OFF and against ON.

**What follows.** `door_loses` or `door_flat` on the target card: the door is deleted in this lane (the list in
`MOE-SLOT-CACHE-DOOR.md` "Decision at decide-by", including every improvement's code that only the door uses), the
receipts banked, the verdict line in the `docs/FLAGS.md` "Removed doors" ledger. `door_wins` on the target card: the
owner's promotion call, with `OWED.md` C2 (PP owner placement, mixed-layout budgets, installer generality, the
refused arms, a serving installer with a serving-shape identity gate) as the promotion work. The RTX 5090's verdict is
reported with it and is a per-card input (the per-hardware rule); it does not veto or carry the target card's.

**What each card can decide.** Each card decides its own verdict; nothing is compared across cards.

## 1a. Addendum, written with the scripts and before any rung cell or deciding cell ran

Scripts: `day51-cell.sh` (cells `hashlock`, `spec`, `decide`), reader `day51-decide.py`. Three points where section 1
did not say enough to be executed, settled here before any result exists. None relaxes a term; two add terms.

- **G1's environment.** Day 18's `hashlock` cell ran with an empty MoE env on the RTX 5090 and
  `MEMRA_MOE_RESIDENT=0` on the target card (`day18-local/run-local.sh`, `pro-single-day18/day18-box.sh`); the
  script takes the same through `D51_MOE_ENV`, so the cell is day 18's unchanged.
- **G2 adds two shapes.** Section 1's G2 is the `spec` run (day 11's `run-spec` shape, default GPU budget, the host
  budget). The cell also runs `spec-pressure` (the same plus `MEMRA_MOE_SLOTS=9986`, the pressure the timing cell
  decides at) and `spec-exact8` (the same plus `--expert-bank-gpu-bytes=6881344`, day 11's eight-slot extreme,
  where every expert is a miss and the prefetch, the in-flight queue and the fill are all under the most pressure).
  G2 passes only if all three pass; each has the section-1 terms plus `[experts-via-tier] installed` present and no
  `fill refused` line.
- **G3's trace term without the stage clock.** `decide`'s ON arm runs without `--expert-bank-stages`, so it prints
  no host demand count; "trace lines equal host demands" is read on every rung cell's door arms (days 43 to 50, all
  with the stage clock), and on `decide` the trace term is `physical_reads` equal to the `hit=false` lines plus at
  least one miss, with no stage line on any ON run (the flag really off).

## 1b. Addendum before any rung or deciding cell ran: I10 joins the ladder

`DAY57.md` (I10, the fill completes inside the install) was found from the day-50 dry check and pre-registered as
its own rung with its own cell (`fillwait`) after day 50's. The tree the deciding cell runs on is the tree after
every rung's verdict, I10's included; nothing else in section 1 moves.

## 2. The binary (filled in before the cell runs)

Named here once every improvement's 5090 verdict is in: the final tree's commit, the list of rungs kept or reverted,
the `run-gen` and `run-spec` SHA-256.

**Named 2026-09-25 01:36Z, before any of this day's three cells ran on either card.** The rungs' RTX 5090 verdicts:
I6 failed its default-budget clause on day 43 and passes it on the tuned tree (`DAY43 RESIDFIX rig=rtx5090
integrity=ok no_regression=PASS`); I9, the fill, I1, I2, I7, I4 and I10 pass (`DAY44` to `DAY50`, `DAY57`); I8 and I5
failed their stage clauses on day 48 and pass as I8f and I5f (`DAY58 SMALLFIX rig=rtx5090 integrity=ok i8f=PASS
i5f=PASS`). None is reverted. The final tree is the lane tip `62e848b1f`, whose `memra-engine` is `7ea765687`'s (the
two later commits touch `memra-server` only). Binaries on the RTX 5090: `run-gen-final` =
`0fbf63285c76b5ddead74f6a8c1b488bbe69d4e6cc9d8398f2bb6203da5a4f03`, `run-spec-final` =
`67286cfa67825cb0a7216203b0f4de8016b7fce289672c0e1fda6ce9ab9fbcb4` (built from the lane tree at `4417bbd1b`, the same
crates). The target card builds `final=62e848b1f` on the box.

## 3. The target card (BOX8, DAY52's final phase; receipts `pro-single-day52/{hashlock,spec,decide}/`)

The box built `final=62e848b1f` (`run-gen-final` `35a64e9e...`, `run-spec-final` `e12bac33...`). G1 (the 27B,
`MEMRA_MOE_RESIDENT=0`) and G2 (the three shapes) ran 02:04Z to 02:05Z; `decide` one collector hold 02:05:05Z to
02:09:34Z, 30 runs, regime (`decide/regime.log`, 250 ms, N=1062) SM 2610 to 2857 MHz, power 86.1 to 210.9 W, 44 to
49 C; collector `--validate` rc=0 for all three. Verbatim (`<cell>/reading.log`):

`DAY51 G1 rig=pro-single door_exit=1 last_line='Error: "experts-via-tier artifact SHA256 mismatch"' door_lines=0 control_exit=0 control_match=True -> PASS`

`DAY51 G2 spec rig=pro-single exit=0 self_consistency_pass=True k_lines=8 k_pass=8 installed=True fill_refused_lines=0 -> PASS`

`DAY51 G2 spec-pressure rig=pro-single exit=0 self_consistency_pass=True k_lines=8 k_pass=8 installed=True fill_refused_lines=0 -> PASS`

`DAY51 G2 spec-exact8 rig=pro-single exit=0 self_consistency_pass=True k_lines=8 k_pass=8 installed=True fill_refused_lines=0 -> PASS`

`DAY51 G2 rig=pro-single -> PASS`

`DAY51 G3 rig=pro-single runs=30 integrity=FAIL failed=o1-on-r1 trace lines 22077 misses 0; ...` (the same term on
all ten ON runs, and no other)

`DAY51 DECIDE rig=pro-single gen-only decode: off=0.311 on=0.277 ref=0.255 (N=10 each) on_minus_off pooled=-0.0345 o1=-0.0340 o2=-0.0350 noise=0.0010 ratio=0.889 (o1 0.891, o2 0.887) -> door_wins`

`DAY51 DECIDE rig=pro-single steady window: off=0.251 on=0.240 ref=0.226 (N=10 each) on_minus_off pooled=-0.0110 o1=-0.0110 o2=-0.0110 noise=0.0013 ratio=0.956 (o1 0.956, o2 0.956) -> door_wins`

`DAY51 READING rig=pro-single on install_s median=9.91 ref_minus_off gen=-0.0560 ref_minus_on gen=-0.0215`

`DAY51 VERDICT rig=pro-single integrity=FAIL -> void`

**Why void.** The one failing term is section 1a's trace term, `physical_reads` equal to the `hit=false` lines
"plus at least one miss". Section 1a was written before I10; I10's own registered clause is that no decode demand
misses the host tier (`DAY57.md` clause (i), `physical_reads=0` on every I10 run), and the final tree carries I10,
so every ON run reads `physical_reads=0`, zero `hit=false` lines and 22,077 trace lines (every demand a host hit).
The two registrations contradict each other; the rule says a failed G3 voids the timing verdict, and the verdict is
void. Every other G3 term holds (tapes, `MATCH`, installer identity, `fill_refused=0`, `physical_reads` equal to the
`hit=false` lines, no stage line on an ON run).

## 1c. The corrected trace term, registered after the void cell and before `decide-b`

The trace term on the ON arm is `physical_reads` equal to the `hit=false` lines and at least one trace line (the
trace is present and consistent); "at least one miss" is dropped because I10's registered clause forbids it. Nothing
else changes: the same cell, arms, orders, N, binary and timing rule, run again as a new hold `decide-b` on each card
(`day51-cell.sh decide-b`, `day51-decide.py decide-b`). Stated plainly: the void cell's timing lines above were seen
before this correction; the correction is confined to the integrity term I10 contradicts, and the verdict is
`decide-b`'s, measured after it.

## 4. `decide-b` on the target card (BOX8; receipts `pro-single-day52/decide-b/`)

One collector hold, 02:12:42Z to 02:17:12Z, 30 runs, `run-gen-final` `35a64e9e...` (`62e848b1f`), the approved
artifact, the runner pinned to 12 cores. Regime (`decide-b/regime.log`, the collector's 250 ms CSV, N=1064): SM 180
to 2872 MHz, power 16.0 to 210.1 W, 37 to 49 C. Collector `--validate` rc=0. Verbatim (`decide-b/reading.log`):

`DAY51 G3 rig=pro-single runs=30 integrity=ok`

`DAY51 DECIDE rig=pro-single gen-only decode: off=0.311 on=0.277 ref=0.255 (N=10 each) on_minus_off pooled=-0.0340 o1=-0.0340 o2=-0.0340 noise=0.0005 ratio=0.891 (o1 0.891, o2 0.891) -> door_wins`

`DAY51 DECIDE rig=pro-single steady window: off=0.251 on=0.240 ref=0.226 (N=10 each) on_minus_off pooled=-0.0110 o1=-0.0110 o2=-0.0110 noise=0.0000 ratio=0.956 (o1 0.956, o2 0.956) -> door_wins`

`DAY51 READING rig=pro-single on install_s median=9.91 ref_minus_off gen=-0.0560 ref_minus_on gen=-0.0220`

`DAY51 VERDICT rig=pro-single integrity=ok -> door_wins`

**The target card's verdict is `door_wins`**: with G1, G2 and G3 green, the tuned door's gen-only decode is 0.277 s
against the naked legacy's 0.311 (ratio 0.891 both orders, day 18's 5.743 before the tuning), and its steady window
0.240 against 0.251. Reported beside it, as registered, and material to the owner's call: REF, the legacy with its
own default-OFF prefetch (`MEMRA_MOE_PREFETCH=1`), is faster than the door on both measures (gen 0.255, 22 ms below
the door; window 0.226), and the door's install takes 9.91 s (the SHA lock, the parallel record pass, the fill). Per
section 1, a win on the target card goes to the owner as the promotion call, with `OWED.md` C2 as the promotion
work; the RTX 5090's `decide-b` is queued behind the card's reset (`rtx5090-fault-20260925/`).

## 5. The RTX 5090's three cells (queue v9, 2026-09-25 23:05Z to 23:38Z; `rtx5090-day51/`)

The RTX 5090 queue v9 (`rtx5090-queue-v9-20260926.sh`) ran these after the rig's reboot wiped the queued binaries in `/tmp`: every binary was rebuilt from its named commit by `c-local-build.sh` in a build worktree under the lane's `target/` (CUDA 13.1, sm_120a; build logs in each cell's `builds/`), behind `/tmp/memra-5090.lock` with the card idle (no compute app) before each hold. A rebuilt binary's hash differs from the one named before the first attempt (the build path is part of the binary); its source tree is the named one. Here `run-gen-final` and `run-spec-final` are built from `62e848b1f` (hashes `c05df689...` and `f4eacc4c...`);
section 2 named the 5090's pair built from `4417bbd1b`, and the two trees' `memra-engine` sources are identical
(`git diff 4417bbd1b 62e848b1f -- crates` touches `crates/memra-server/src/worker.rs` only). Regime over the
`decide-b` hold (`command.gpu.csv`): 58 to 68 C, SM median 1612 MHz, N=1173. Verbatim:

- `DAY51 G1 rig=rtx5090 door_exit=1 last_line='Error: "experts-via-tier artifact SHA256 mismatch"' door_lines=0 control_exit=0 control_match=True -> PASS`
- `DAY51 G2 rig=rtx5090 -> PASS` (`spec`, `spec-pressure` and `spec-exact8` each `k_pass=8`)
- `DAY51 G3 rig=rtx5090 runs=30 integrity=ok`
- `DAY51 DECIDE rig=rtx5090 gen-only decode: off=0.411 on=0.375 ref=0.365 (N=10 each) on_minus_off pooled=-0.0360 o1=-0.0360 o2=-0.0240 noise=0.0260 ratio=0.912 (o1 0.912, o2 0.942) -> door_flat`
- `DAY51 DECIDE rig=rtx5090 steady window: off=0.409 on=0.405 ref=0.401 (N=10 each) on_minus_off pooled=-0.0040 o1=-0.0020 o2=-0.0080 noise=0.0072 ratio=0.990 (o1 0.995, o2 0.981) -> door_flat`
- `DAY51 READING rig=rtx5090 on install_s median=11.06 ref_minus_off gen=-0.0460 ref_minus_on gen=-0.0100`
- `DAY51 VERDICT rig=rtx5090 integrity=ok -> door_flat`

**Read as registered: G1 and G2 PASS, `door_flat` on the RTX 5090.** The door's medians sit below the naked legacy
(0.375 against 0.411 gen-only) but inside this card's noise (0.026), so the rule reads `flat`. Per section 1 the
5090's verdict is a per-card input and does not veto or carry the target card's `door_wins`. REF (the legacy with
`MEMRA_MOE_PREFETCH=1`) is below the door here too: 0.365 gen-only, 0.401 window. The door's install takes 11.06 s.
