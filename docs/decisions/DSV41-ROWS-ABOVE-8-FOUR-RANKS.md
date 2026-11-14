# DeepSeek-V4.1 on four ranks: no call above 8 rows by default until memra#1311 is fixed (2026-11-14)

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
