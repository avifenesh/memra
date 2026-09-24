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
