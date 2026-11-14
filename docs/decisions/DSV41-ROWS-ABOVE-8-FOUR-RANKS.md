# DeepSeek-V4.1 on four ranks: no call above 8 rows by default until memra#1311 is fixed (2026-11-14; amended the same day: the fix)

> Amended 2026-11-14: memra#1311's fix changes the split MoE's gather rule
> ([amendment](#amendment-2026-11-14-the-gather-fix-measured-in-two-halves-its-confirm-pending)).
> Window 3 measured it on four cards in two halves; the final binary's own four-card confirm is
> pending. Until that confirm is in the tree, the decision below stands as written.

## Decision

The CUTLASS lane of the deepseek-v4.1 route (memra#1286) lands with a known loss on four ranks at
calls above 8 rows. Two limits keep served calls off those widths, and both stay where they are
on a four-rank split until memra#1311 is fixed or re-measured:

- **The served batch cap stays at 8 rows.** `MEMRA_DSV41_BATCH_ROWS` is 8 at most
  (`BATCH_ROWS_MAX` in `crates/memra-server/src/dsv41_serve.rs`, the small-M bodies' limit), so
  a plain served decode step never issues a call above 8 rows.
- **Drafting stays off.** `MEMRA_DSV41_DRAFT=1` puts each drafting request's next token and its
  drafted ones in the step's forward, so two requests that draft can pass 8 rows together
  (`MEMRA_DSV41_DRAFT_DEPTH` caps a request's share at the head's block of 5 plus one). Its
  default is off for its own reasons (its served gates have not run). Its interleaved A/B on a
  four-rank split has to run on a build with memra#1311 fixed, or has to measure these widths
  again.

Raising the cap above 8 rows, or turning drafting on, on a split of more than two ranks needs
that issue closed or a new four-card measurement first. One card and two ranks are not covered
by this record: the pair is faster at every width from 8 rows.

## The measurement

Four RTX PRO 6000 Blackwell Max-Q, 40 layers, the default transport, 2026-11-14, nothing else on
the box. A is main `8031c7ef90` (the route before the lane), B is `a05173ee27` (the lane's
head). A B B A B A A B, four runs an arm, steady calls past 0.5 s of load, ms; the range is
B / A of the four adjacent pairs of runs:

| call | A | B | B / A |
|---|---|---|---|
| one-row decode | 9.893 | 9.712 | 0.982 (0.981-0.982) |
| 4 rows | 12.940 | 13.059 | 1.009 (1.007-1.010) |
| 8 rows | 16.926 | 16.840 | 0.995 (0.994-0.995) |
| 9 rows | 19.009 | 19.179 | 1.009 (1.008-1.010) |
| 12 rows | 20.686 | 22.091 | 1.068 (1.066-1.069) |
| 16 rows | 24.000 | 24.857 | 1.036 (1.034-1.037) |

The pair in the same window (20 layers): 0.983, 0.976 and 0.952 at 9, 12 and 16 rows.

Raw runs: `research/modelplan-onboarding-dsv41-20261109/perf/receipts/cfix-pro6000/cfix2-four`
and `cfix2-two`. The lane's README has the full write-up
(`research/modelplan-onboarding-dsv41-20261109/perf/README.md`, "Window 2 (2026-11-14)").

## Why it is slower

Above 8 rows the base gathers the split MoE's hidden rows as FP8 codes and scales. The lane
keeps one bf16 gather through 16 rows: one gather fewer a layer, twice the bytes. A gather stays
inside kernels while a rank stores at most 224 KiB, its region times its peers. On four ranks
the bf16 rows pass that at 10 rows (236.2 KiB), so every layer of a 12- or 16-row call runs a
packed gather with 12 peer copies where the base runs two pushes. The host's issue time is level
at 8 and 9 rows and 1.1 ms longer at 12 and 16. On two ranks the same gather is a push through
14 rows.

The run data does not explain the +0.9% at 9 rows on four cards or the +0.9% at 4 rows (also on
the pair): `dsv41-tpbench` records no per-launch timing.

## What was rejected

- **Holding the lane for the fix.** The lane is faster at one row and at 8 rows on four cards,
  faster on the pair at every width but 4 rows, and 15 to 24% faster at prefill. The widths that
  lost are not served by default.
- **Changing the split MoE's gather rule inside the landing.** A rule that takes the codes
  gather above 8 rows on more than two ranks is the candidate fix. It changes which gathers a
  four-rank call issues, so it needs its own four-card A/B and lands as its own change.

## Amendment 2026-11-14: the gather fix, measured in two halves, its confirm pending

**The rule.** `split_quant` (`dsv41_gpu/tp.rs`): on more than two ranks the split MoE keeps the
bf16 gather of its hidden rows while a rank stores at most 224 KiB (its region times its peers),
and takes the codes and scales when it would not. On four ranks that is bf16 through 9 rows and
codes from 10. Two ranks keep bf16 through 16 rows.

**What window 3 measured** (four RTX PRO 6000 Blackwell Max-Q, 40 layers, quiet box, A B B A B A
A B, four runs an arm; `gather1` is `c99798ef57`, the first form of the fix: codes above 8 rows
on more than two ranks). B / A, with the four pairs of runs' range:

| call | `gather1` against main `63c351a2f4` | `gather1` against the base `8031c7ef90` |
|---|---|---|
| one-row decode | 1.000 (1.000-1.002) | 0.981 (0.979-0.981) |
| 4 rows | 1.000 (0.998-1.001) | 1.002 (1.000-1.004) |
| 8 rows | 1.000 (1.000-1.001) | 0.997 (0.995-0.999) |
| 9 rows | 1.011 (1.010-1.013) | 1.022 (1.019-1.023) |
| 10 rows | 0.949 (0.947-0.951) | 1.010 (1.008-1.012) |
| 12 rows | 0.946 (0.943-0.947) | 1.001 (0.998-1.002) |
| 16 rows | 0.955 (0.955-0.958) | 0.987 (0.986-0.988) |

10, 12 and 16 rows are fixed: 12 rows is level with the base (was +6.8%) and 16 rows is 1.3%
faster (was +3.6%). 9 rows got 1.1% slower than main, because the bf16 gather is still a push
there. So the rule moved to the push limit after the window.

**What the final binary is.** At 4, 8 and 9 rows its four-rank program is main's (arm A of the
first block). At 10 rows and above it is `gather1`'s (arm B). **The final binary itself has no
four-card run**: its two halves were each measured, in one window. Its four-card confirm against
main (4, 8, 9, 10, 12 and 16 rows, the kernel and tp gates first) is pending and is required
before the fix lands. On one GB10 (four ranks on the card, 4 layers) the final build's own gather
counters show the bf16 gather at 4, 8 and 9 rows and the codes from 10: one collective more a
layer and 0.774 of the bytes a row, the step between window 3's two arms.

**What the decision becomes when the confirm is in.** The loss this record was written for (12
and 16 rows) is gone, so the gather no longer holds the batch cap or drafting on a four-rank
split. What remains against the base, unexplained (memra#1324):

- 9 rows: +0.9% (Window 2; the fix does not change that program);
- 10 rows: +1.0% (window 3);
- 4 rows: +0.9% in Window 2, +0.2% in window 3.

With the base's own gathers this tree is 2.2% behind at 9 rows and 1.0% at 10, so the remainder
is in the kernels or the launches, not the gathers. A change that makes calls of 9 or 10 rows the
common case on four ranks (a batch cap above 8, drafting on by default) reads that issue first.
Drafting stays off for its own reasons: its served gates have not run.

**Rejected in the fix.** Codes above 8 rows on every split of more than two ranks (`gather1`):
it fixes 10 to 16 rows and costs 1.1% at 9 rows. Raw runs:
`research/modelplan-onboarding-dsv41-20261109/perf/receipts/cfix-pro6000/gather1-four` and
`gather1-base-four`.
