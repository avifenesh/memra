# The stall mark: the spec cost model sees speculation that keeps meeting waiting requests (memra#1211)

Date: 2026-11-11. Qwen3.8-27B NVFP4 + Q5_K with its MTP head, context 8,192, eight sessions,
every switch at its default unless a row says otherwise. Pass/fail style rate cells: single runs
per boot, no timing claim beyond the rows. Decision record:
[`docs/decisions/SPEC-STALL-MARK.md`](../../docs/decisions/SPEC-STALL-MARK.md). Follows
`research/spec-entry-yield-bound-20261110/` (memra#1201), whose slot arm this replaces, and
`research/prime-hold-tick-skip-20261111/` (memra#1223), which it builds on.

**Verdict: adopted, the stall mark with its wave hold and its refresh (memra#1211).** On GB10 at
the default gate, when a speculating session's last four burst entries each met a waiting
request, the cost model marks that concurrency and the waves its newcomers are admitted into
plain for 64 s; the session the mark demoted holds its waves and refreshes the mark while
requests keep waiting. Against main 98337756cf on one unit the long stream gets 155 to 226 tokens in
30 s where it got 49 to 141. Short p50 falls to 0.30 to 0.39 s from 0.42 to 0.82 in every
marked cell but one: at two clients and 200 ms think it is a trade (+60% long tokens for +11 ms
short p50). The gain holds
through 120 s of arrivals, admissions return to the boot price 64 s after the pressure stops,
and no mark is set at one client, at 1000 ms think or at c = 1..8. A mark is not a price: no
realized slot changes and nothing scales from it. Texts are unchanged. Rejected on measurement:
a mark at the tick's count, the tick's count plus the number waiting, the admission mark without
the hold, the hold without the refresh. memra#1211's share questions (how many rounds a
speculating stream keeps, whether a one-token request takes the speculative path) stay open.

## Conditions

- **GB10**: one DGX Spark unit per boot, `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`, the calibrated cost
  model owns the gate. One boot per lease. `MEMRA_TICK_TRACE=1` where a row's `extra_env` says so.
- **RTX 5090 Laptop GPU**: the same 27B (its calibration is refused on this card, memra#1191, so
  the band owns the gate) and Ornith-1.5-35B-A3B NVFP4 + Q5_K MTP (the cost model owns it,
  unpooled).
- **Binaries** (GB10 `121a`, the 5090 `120a`):

  | arm | tree | what it is |
  |---|---|---|
  | `main` | 83d37f5179 (crates equal main 66ee31fcb6) | main before memra#1228, the main arm of the first two forms |
  | `change` | 052bbc0e23 | first form: a mark at the tick's count |
  | `change2` | dd243cf79c | second form: the tick's count and the count plus the number waiting |
  | `grace2` (`grace` on the 5090) | 7c8028cd17 on ede1b8119f, `memra-0.141.1-b87b16188411` | main with memra#1228: the main arm of every bar below |
  | `f1b` | d09028eb17, `memra-0.141.1-1fcf21cf875b` | the exact form: the tick's count and the admitted wave |
  | `f1c` | d45ece6446, `memra-0.141.1-9ed89dd36d80` | the exact form and the wave hold |
  | `f1d` | 1d76cd2c08, `memra-0.141.1-fbdc32f79efb` | the shipped mechanism: the exact form, the hold and the refresh |
  | `main7` | main 98337756cf, `memra-0.141.1-91a9b8877403` | main after landing memra#1228, memra#1186 (a sampled burst carries its pending token), memra#1209 and memra#1217 |
  | `head` | the shipped branch: f1d rebased on 98337756cf, with the review's fixes, `memra-0.141.1-4aa0082c93e2` | what lands |

- **Units.** The two Spark units differ by a few ms and tokens in cells with no mark at all
  (`gb10/sweep-exact/`: grace2 on one unit against f1b on the other, then grace2 again on the
  second unit matching f1b). Every arm pair compared below ran on one unit unless a section
  says otherwise.
- **Cells** (`tools/stall_cell.py`): a long greedy streamed request runs alone for 8 s, then
  closed-loop clients send one-token chat requests for 30 s (or 120 s), each waiting the think
  time after every answer, then the long request runs alone for 12 s. A cell is written
  `clients:think ms`. A mark lives 64 s, so cells are 70 s apart. Each row carries the census:
  marks set, stalled verdicts, refreshes, `[spec-k]` admissions by K and source, demotions by
  reason. A sixth cell field `live` keeps the long request open through a follow phase (`c/gap`:
  after the tail and `gap` seconds, `c` clients send three greedy 160-token requests each), so
  the follow's admissions see the demoted session still live. `tools/normal_cell.py`: c = 1, 2,
  4, 8 closed-loop clients, three greedy 160-token streamed requests each.
- **The bar** (per cell, form against main with memra#1228): the long stream advances at least
  as much; at least 90% of newcomers are admitted plain in every cell that marks; short p50 and
  p95 no worse than main's, or the exception stated as a measured trade; 0 marks and identical
  choices in the 1-client, 1000 ms and c = 1..8 cells; the gain holds through 120 s of arrivals;
  admissions return to the boot price within 64 s after the pressure stops.
  `tools/bar1211.py` prints the per-cell items.

## The forms, in the order they were built

1. **A mark at the tick's count** (`change`). Four pressed burst entries in a row mark the
   tick's live count.
2. **The tick's count plus the number waiting** (`change2`).
3. **The exact form** (`f1b`): the tick's count, and each request waiting at the run's last
   entry marks, at its own admission, the wave its verdict reads. Unused admission marks lapse
   at the next tick.
4. **The wave hold** (`f1c`): a session demoted by a live mark keeps that mark's count; before
   every admission the hold is the latest expiry among those sessions' live marks, and the
   admission marks its own wave until that expiry, never on a fresh time-to-live.
5. **The refresh** (`f1d`, shipped): a session demoted by a live mark refreshes it when it steps
   its plain decode while a request waits for admission. Only a live mark is refreshed, never
   created.

## The first two forms (`gb10/first-forms/`)

One boot of `main`, two of `change`, one of `change2`, before memra#1228.

`change` against `main`:

| cell | long tokens main / form | short p50 s main / form | short p95 s main / form | marks form | plain/all form | items |
|---|---|---|---|---|---|---|
| 1:0 | 230 / 230 | 0.309 / 0.309-0.31 | 0.312 / 0.31-0.312 | 0 | 0/98 0/98 | ok |
| 2:0 | 77 / 191-193 | 0.479 / 0.3-0.301 | 0.519 / 0.304-0.311 | 1 | 186/194 188/196 | ok |
| 4:0 | 56 / 153 | 0.833 / 0.358-0.359 | 0.933 / 0.696-0.73 | 1 | 296/312 296/312 | ok |
| 1:200 | 424 / 427-435 | 0.306 / 0.305 | 0.309 / 0.311-0.312 | 0 | 0/61 0/60 | 3 p95 WORSE |
| 2:200 | 142 / 153-154 | 0.318 / 0.346-0.349 | 0.323 / 0.549-0.55 | 1 | 25/103 25/104 | 3 p50 WORSE, 3 p95 WORSE, 2 plain 24% |
| 4:200 | 50 / 92 | 0.729 / 0.637 | 0.744 / 0.952-0.954 | 1 | 54/152 54/152 | 3 p95 WORSE, 2 plain 36% |
| 1:1000 | 632 / 631 | 0.298 / 0.292-0.298 | 0.309 / 0.302-0.306 | 0 | 0/24 0/24 | 1 WORSE |
| 2:1000 | 499 / 499-501 | 0.221 / 0.222-0.302 | 0.312 / 0.311-0.315 | 0 | 0/48 0/48 | 3 p50 WORSE |
| 4:1000 | 229 / 229 | 0.236 / 0.233-0.236 | 0.331 / 0.329-0.334 | 0 | 0/98 0/98 | ok |
| 2:0 t1 | 63 / 191-193 | 0.491 / 0.301-0.302 | 0.494 / 0.306-0.312 | 1 | 186/194 188/196 | ok |
| 4:0 t1 | 35 / 151-153 | 0.906 / 0.36 | 0.918 / 0.723-0.789 | 1 | 296/312 292/308 | ok |

`bar1211.py`: items to explain: 1:200: 3 p95 WORSE; 2:200: 3 p50 WORSE; 2:200: 3 p95 WORSE; 2:200: 2 plain 24%; 4:200: 3 p95 WORSE; 4:200: 2 plain 36%; 1:1000: 1 WORSE; 2:1000: 3 p50 WORSE

`change2` against `main`:

| cell | long tokens main / form | short p50 s main / form | short p95 s main / form | marks form | plain/all form | items |
|---|---|---|---|---|---|---|
| 1:0 | 230 / 230 | 0.309 / 0.309 | 0.312 / 0.311 | 0 | 0/98 | ok |
| 2:0 | 77 / 193 | 0.479 / 0.301 | 0.519 / 0.307 | 1 | 188/196 | ok |
| 4:0 | 56 / 153 | 0.833 / 0.358 | 0.933 / 0.695 | 1 | 296/312 | ok |
| 1:200 | 424 / 435 | 0.306 / 0.305 | 0.309 / 0.308 | 0 | 0/61 | ok |
| 2:200 | 142 / 261 | 0.318 / 0.473 | 0.323 / 0.483 | 1 | 86/91 | 3 p50 WORSE, 3 p95 WORSE |
| 4:200 | 50 / 92 | 0.729 / 0.637 | 0.744 / 0.949 | 1 | 54/152 | 3 p95 WORSE, 2 plain 36% |
| 1:1000 | 632 / 631 | 0.298 / 0.294 | 0.309 / 0.3 | 0 | 0/24 | 1 WORSE |
| 2:1000 | 499 / 505 | 0.221 / 0.305 | 0.312 / 0.319 | 0 | 0/48 | 3 p50 WORSE, 3 p95 WORSE |
| 4:1000 | 229 / 229 | 0.236 / 0.232 | 0.331 / 0.326 | 0 | 0/98 | ok |
| 2:0 t1 | 63 / 193 | 0.491 / 0.301 | 0.494 / 0.312 | 1 | 188/196 | ok |
| 4:0 t1 | 35 / 153 | 0.906 / 0.359 | 0.918 / 0.706 | 1 | 296/312 | ok |

`bar1211.py`: items to explain: 2:200: 3 p50 WORSE; 2:200: 3 p95 WORSE; 4:200: 3 p95 WORSE; 4:200: 2 plain 36%; 1:1000: 1 WORSE; 2:1000: 3 p50 WORSE; 2:1000: 3 p95 WORSE

- Both fix 2:0 and 4:0 and the sampled cells (the long stream at 153 to 193 tokens against 35 to
  77, 95% of newcomers plain).
- `change` fails at 200 ms think: the mark lands at the tick's count (n=2 at 2:200) while
  newcomers are priced at the projected wave (n=3), so 25 of 103 stay speculative and short p95
  is 0.55 s against 0.32.
- `change2` fails at 4:200: requests that finish inside the tick leave room, so the mark names
  n=4 and n=7 while admissions run at wave 5 (54 of 152 plain).
- `change2` at 2:200 admits 86 of 91 plain but short p50 is 0.473 s against 0.318. The tick
  trace showed a different defect: a lone plain arrival lost whole ticks to the 4 ms prime holds
  (memra#1223, fixed for the formation hold by memra#1228; its receipts are in
  `research/prime-hold-tick-skip-20261111/gb10/diag/`). Every later form is measured on main
  with that fix.
- 2:1000 short p50 sits on one of two modes (0.22 or 0.30 s) by arrival phase on both arms, with
  0 marks and the same admissions.

## The exact form (`gb10/sweep-exact/`)

ABBA: grace2-31 (one unit), f1b-32, f1b-33, grace2-34 (the other unit). grace2-34 matches f1b
in every cell with no mark (1:1000 p50 0.302 s and 630 tokens against 0.297-0.301 s and 631),
so the units, not the code, made the 0-mark differences of the first pair.

| cell | long tokens main / form | short p50 s main / form | short p95 s main / form | marks form | plain/all form | items |
|---|---|---|---|---|---|---|
| 1:0 | 230-237 / 230 | 0.305-0.309 / 0.308-0.309 | 0.308-0.315 / 0.311-0.316 | 0 | 0/98 0/98 | ok |
| 2:0 | 77-78 / 193 | 0.422-0.483 / 0.3-0.301 | 0.513-0.519 / 0.304-0.309 | 1 | 188/196 188/196 | ok |
| 4:0 | 56-58 / 153 | 0.823-0.835 / 0.357-0.358 | 0.923-0.94 / 0.707-0.713 | 1 | 296/312 296/312 | ok |
| 1:200 | 427-453 / 431-439 | 0.299-0.307 / 0.305-0.306 | 0.303-0.311 / 0.313 | 0 | 0/60 0/60 | 3 p95 WORSE |
| 2:200 | 142-143 / 135-137 | 0.312-0.319 / 0.317 | 0.317-0.328 / 0.477-0.478 | 1 | 26/109 27/110 | 1 WORSE, 3 p95 WORSE, 2 plain 24% |
| 4:200 | 50-53 / 91-92 | 0.72-0.731 / 0.637-0.64 | 0.735-0.748 / 0.952-0.954 | 1 | 51/151 54/152 | 3 p95 WORSE, 2 plain 34% |
| 1:1000 | 630-639 / 631 | 0.277-0.302 / 0.297-0.301 | 0.282-0.308 / 0.305-0.307 | 0 | 0/24 0/24 | ok |
| 2:1000 | 490-515 / 499 | 0.22-0.295 / 0.217-0.228 | 0.301-0.313 / 0.31-0.311 | 0 | 0/48 0/49 | ok |
| 4:1000 | 228-233 / 229 | 0.224-0.239 / 0.234-0.237 | 0.317-0.336 / 0.328-0.334 | 0 | 0/98 0/98 | ok |
| 2:0 t1 | 63-65 / 191-193 | 0.487-0.493 / 0.301-0.302 | 0.494-0.5 / 0.312-0.313 | 1 | 186/194 188/196 | ok |
| 4:0 t1 | 35-37 / 153 | 0.898-0.908 / 0.359-0.36 | 0.903-0.918 / 0.718-0.729 | 1 | 296/312 296/312 | ok |

`bar1211.py`: items to explain: 1:200: 3 p95 WORSE; 2:200: 1 WORSE; 2:200: 3 p95 WORSE; 2:200: 2 plain 24%; 4:200: 3 p95 WORSE; 4:200: 2 plain 34%

It fails the plain-admission item at 2:200 (26 of 109 newcomers plain) and 4:200 (51 of 151).
The `[spec-k]` lines per admission wave show why: at 200 ms think the short clients overlap, so
admissions alternate between wave n and n+1. A single stall marks n (2:200: n=2; 4:200: n=4);
26 admissions at wave 2 go plain and 80 at wave 3 stay speculative (4:200: 51 at wave 4, 89 at
wave 5). Nothing marks n+1: the demoted long stream is plain and never bursts again, one-token
requests never burst, and at 200 ms think no request is waiting when the stall is marked, so
the admission marks lapse.

## The wave hold (`gb10/sweep/`)

ABBA on one unit: grace2-41, f1c-42, f1c-43, grace2-44.

| cell | long tokens main / form | short p50 s main / form | short p95 s main / form | marks form | plain/all form | items |
|---|---|---|---|---|---|---|
| 1:0 | 230 / 230 | 0.307-0.309 / 0.309 | 0.314-0.315 / 0.313 | 0 | 0/98 0/98 | ok |
| 2:0 | 77-78 / 193 | 0.467-0.494 / 0.301 | 0.516-0.517 / 0.304-0.31 | 1 | 188/196 188/196 | ok |
| 4:0 | 56 / 153 | 0.83-0.831 / 0.358 | 0.933-0.937 / 0.707-0.719 | 1 | 296/312 296/312 | ok |
| 1:200 | 431-439 / 424-435 | 0.304-0.306 / 0.306 | 0.311-0.313 / 0.312 | 0 | 0/61 0/60 | ok |
| 2:200 | 142 / 220-223 | 0.317-0.318 / 0.33-0.331 | 0.324-0.325 / 0.337-0.341 | 2 | 108/113 109/114 | 3 p50 WORSE, 3 p95 WORSE |
| 4:200 | 50 / 187-188 | 0.723-0.728 / 0.394-0.396 | 0.739-0.741 / 0.63-0.632 | 2 | 184/197 184/197 | ok |
| 1:1000 | 631 / 630-631 | 0.291-0.295 / 0.296-0.3 | 0.297-0.301 / 0.304 | 0 | 0/24 0/24 | 3 p50 WORSE, 3 p95 WORSE |
| 2:1000 | 501-509 / 494-499 | 0.221-0.301 / 0.222-0.226 | 0.312-0.315 / 0.308-0.31 | 0 | 0/48 0/49 | 1 WORSE |
| 4:1000 | 229 / 229 | 0.231-0.233 / 0.233-0.234 | 0.324-0.333 / 0.332-0.335 | 0 | 0/98 0/98 | ok |
| 2:0 t1 | 62-64 / 193 | 0.489-0.491 / 0.301 | 0.497 / 0.304-0.311 | 1 | 188/196 188/196 | ok |
| 4:0 t1 | 35-36 / 153 | 0.902-0.905 / 0.359 | 0.909-0.915 / 0.72-0.723 | 1 | 296/312 296/312 | ok |

`bar1211.py`: items to explain: 2:200: 3 p50 WORSE; 2:200: 3 p95 WORSE; 1:1000: 3 p50 WORSE; 1:1000: 3 p95 WORSE; 2:1000: 1 WORSE

- **2:200 and 4:200** now admit 96% and 93% of newcomers plain (108 of 113, 184 of 197). The
  hold's mark line names it: `stalled n=3: the wave holds a session a stall mark demoted; plain
  at n=3 while that mark lives (63.7 s)`.
- **2:200 short latency is a measured trade, not a pass.** The long stream gets 220-223 tokens
  against 142 (+55%); short p50 is 0.330-0.331 s against 0.317-0.318 (+13 ms) and p95
  0.337-0.341 s against 0.324-0.325 (+13 to 16 ms). The tick trace (`tools/tickparts.py`, the
  2:200 cell) says where it goes:

  | arm | ticks that primed a short | prefill ms | decode ms | formation wait ms | tick sum ms |
  |---|---:|---:|---:|---:|---:|
  | f1c-42 | 108 | 112.9 | 78.0 | 3.98 | 194.8 |
  | f1c-43 | 109 | 112.6 | 77.6 | 3.98 | 194.2 |

  Every plain short primes in its admission tick (108 of 108, 109 of 109). On main the shorts
  are speculative and prime inside bursts of about 306 ms that give their tick back to an
  arrival at the burst entry (131 and 132 ticks, no prefill or decode phase). With the mark a
  short arriving mid-tick waits for the long stream's plain decode step (78 ms), which never
  yields, and each lone plain prime pays memra#1228's 4 ms formation wait (107 and 108 per
  cell). The formation wait accounts for about 4 ms of the 13; the rest is structural to a
  plain long stream's decode step.
- **4:200** passes outright: long 187-188 against 50, p50 0.394-0.396 s against 0.723-0.728,
  p95 0.630-0.632 s against 0.739-0.741.
- **0-mark cells** (1-client, 1000 ms): the same draft length at every admission on both arms.
  The remaining flags are run spread on one unit: 1:1000 p50 0.296-0.300 s against 0.291-0.295
  (24 requests per cell), 2:1000 long 494-499 against 501-509, where main's own two boots
  differ by 8 tokens and 80 ms of p50.
- Every long text is prefix-equal to main's (`tools/prefix1211.py`).

## 120 s of arrivals: why the refresh (`gb10/long-hold/`, `gb10/long/`, `gb10/long-same-unit/`)

First measurement (grace2 and f1c, then f1d; the units and the trace setting differ between
these boots, see each row's `extra_env` and `guard.log`):

| arm | cell | long tokens per 30 s segment | total | short p50 s per segment | short p50 / p95 s, whole | newcomers plain / all | marks | refreshes |
|---|---|---|---:|---|---|---|---:|---:|
| grace2 | 2:0 | 77 / 81 / 72 / 84 | 315 | 0.424 / 0.509 / 0.416 / 0.508 | 0.437 / 0.515 | 0 / 520 | 0 | 0 |
| grace2 | 4:0 | 56 / 168 / 168 / 168 | 562 | 0.827 / 0.356 / 0.357 / 0.357 | 0.357 / 0.866 | 1052 / 1172 | 0 | 0 |
| f1c | 2:0 | 196 / 201 / 91 / 63 | 552 | 0.297 / 0.297 / 0.48 / 0.481 | 0.298 / 0.482 | 432 / 666 | 1 | - |
| f1c | 4:0 | 153 / 170 / 68 / 34 | 426 | 0.354 / 0.355 / 0.888 / 0.892 | 0.355 / 0.893 | 724 / 978 | 1 | - |
| f1d | 2:0 | 196 / 201 / 202 / 201 | 801 | 0.297 / 0.298 / 0.298 / 0.298 | 0.298 / 0.3 | 794 / 802 | 1 | 396 |
| f1d | 4:0 | 156 / 166 / 166 / 168 | 658 | 0.354 / 0.356 / 0.356 / 0.356 | 0.356 / 0.381 | 1300 / 1316 | 1 | 324 |

- **The hold alone loses the gain 64 s after the stall.** The demoted stream is plain and never
  enters a burst again, so nothing refreshes the mark; when it expires, newcomers are admitted
  speculatively again and the cell falls back to main's level (2:0) or below it (4:0: 68 and 34
  long tokens against 168, short p50 0.89 s against 0.36).
- **With the refresh the gain holds in every segment** (2:0: 196 / 201 / 202 / 201; 396
  refreshes for 1 mark).

The same two cells for both arms on one unit with the same env (`gb10/long-same-unit/`):

| arm | cell | long tokens per 30 s segment | total | short p50 s per segment | short p50 / p95 s, whole | newcomers plain / all | marks | refreshes |
|---|---|---|---:|---|---|---|---:|---:|
| grace2 | 2:0 | 77 / 81 / 72 / 84 | 314 | 0.424 / 0.424 / 0.509 / 0.425 | 0.425 / 0.517 | 0 / 519 | 0 | 0 |
| grace2 | 4:0 | 56 / 168 / 168 / 167 | 560 | 0.827 / 0.357 / 0.357 / 0.357 | 0.357 / 0.865 | 1048 / 1168 | 0 | 0 |
| f1d | 2:0 | 195 / 201 / 201 / 201 | 799 | 0.298 / 0.298 / 0.298 / 0.299 | 0.298 / 0.3 | 792 / 800 | 1 | 395 |
| f1d | 4:0 | 155 / 169 / 169 / 168 | 663 | 0.354 / 0.356 / 0.356 / 0.356 | 0.356 / 0.36 | 1312 / 1328 | 1 | 327 |

On one unit with the same env the 4:0 cell is a tie after the first segment (169 / 169 / 168 against 168 / 168 / 167) and the gain is the first segment (155 against 56) and the whole-cell short p95 (0.360 s against 0.865). At 2:0 the gain holds in all four segments.

## Recovery with the long stream still open (`gb10/recovery-hold/`, `gb10/recovery/`)

A pressure cell, then a c=2 follow of six 160-token requests while the long request (2,000
tokens) is still open.

| arm | pressure cell | follow starts after s | window marks | window newcomers plain / all | long open through the follow | follow K at admission | follow refreshes | follow agg tok/s | long tokens in the follow | follow texts equal across arms |
|---|---|---:|---:|---|---|---|---:|---:|---:|---|
| grace2 | 2:0 | 5.0 | 0 | 0 / 131 | yes | K3/cost-model x6 | - | 29.98 | 595 | 6 of 6 |
| grace2 | 2:0 | 70.0 | 0 | 0 / 131 | yes | K3/cost-model x6 | - | 31.03 | 588 | 6 of 6 |
| grace2 | 2:200 | 70.0 | 0 | 0 / 118 | True / False | K3/cost-model x6 | - | 31.73 | 538 | 6 of 6 |
| f1c | 2:0 | 5.0 | 1 | 192 / 200 | yes | K0/cost-model x4 K3/cost-model x2 | - | 29.04 | 391 | 6 of 6 |
| f1c | 2:0 | 70.0 | 1 | 192 / 200 | yes | K3/cost-model x6 | - | 39.82 | 197 | 6 of 6 |
| f1c | 2:200 | 70.0 | 2 | 110 / 116 | yes | K3/cost-model x6 | - | 39.75 | 197 | 6 of 6 |
| f1d | 2:0 | 5.0 | 1 | 190 / 198 | yes | K0/cost-model x6 | 3 | 25.0 | 486 | 6 of 6 |
| f1d | 2:0 | 70.0 | 1 | 188 / 196 | yes | K3/cost-model x6 | 0 | 38.94 | 197 | 6 of 6 |
| f1d | 2:200 | 70.0 | 2 | 110 / 115 | yes | K3/cost-model x6 | 0 | 38.84 | 197 | 6 of 6 |

- **Inside the time-to-live the marked waves serve plain.** f1d's follow 5 s after 2:0 admits
  all six plain (the demoted stream's decode steps refresh the mark three times while follow
  requests wait); its aggregate is 25.0 tok/s against main's 29.98. That is the cost of a
  verdict that outlives the pressure, bounded by 64 s of quiet or the demoted session's end.
  On f1c the mark expired mid-follow (K=0 x4, then K=3 x2).
- **After 70 s of quiet admissions are K=3 again, priced as at boot**, although the demoted
  stream is still open: its mark expired, so the session dropped its count and holds nothing.
  The follow's aggregate is then higher than main's (38.8 to 39.8 against 31.0 to 31.7 tok/s)
  because the long stream beside it is plain and steps in the batched phase.
- Follow texts are equal across arms in all nine follows.

## Ordinary concurrency (`gb10/normal-hold/`, `gb10/normal/`)

c = 1, 2, 4, 8, three boots each of f1c and f1d. The main arm is memra#1228's census
(`research/prime-hold-tick-skip-20261111/gb10/ab-normal/`: main2-21, main2-24 before it,
grace2-22, grace2-23 with it); texts are compared with its first boot.

| boot | c | agg tok/s | TTFT p50 s | K at admission | marks | stalled verdicts | texts equal to main2-21 |
|---|---:|---:|---:|---|---:|---:|---|
| main2-21 | 1 | 24.7 | 0.301 | K3/cost-model x3 | 0 | 0 | 3 of 3 |
| main2-21 | 2 | 39.93 | 0.462 | K3/cost-model x6 | 0 | 0 | 6 of 6 |
| main2-21 | 4 | 52.98 | 0.914 | K3/cost-model x12 | 0 | 0 | 12 of 12 |
| main2-21 | 8 | 79.95 | 1.521 | K0/cost-model x22 K1/resume-floor(k=0->1) x1 K3/cost-model x1 | 0 | 0 | 24 of 24 |
| grace2-22 | 1 | 24.68 | 0.298 | K3/cost-model x3 | 0 | 0 | 3 of 3 |
| grace2-22 | 2 | 35.93 | 0.302 | K3/cost-model x6 | 0 | 0 | 6 of 6 |
| grace2-22 | 4 | 53.05 | 0.908 | K3/cost-model x12 | 0 | 0 | 12 of 12 |
| grace2-22 | 8 | 79.86 | 1.416 | K0/cost-model x23 K3/cost-model x1 | 0 | 0 | 24 of 24 |
| grace2-23 | 1 | 24.68 | 0.298 | K3/cost-model x3 | 0 | 0 | 3 of 3 |
| grace2-23 | 2 | 39.97 | 0.459 | K3/cost-model x6 | 0 | 0 | 6 of 6 |
| grace2-23 | 4 | 53.81 | 0.917 | K3/cost-model x12 | 0 | 0 | 12 of 12 |
| grace2-23 | 8 | 79.77 | 1.464 | K0/cost-model x23 K3/cost-model x1 | 0 | 0 | 24 of 24 |
| main2-24 | 1 | 24.66 | 0.298 | K3/cost-model x3 | 0 | 0 | 3 of 3 |
| main2-24 | 2 | 40.02 | 0.457 | K3/cost-model x6 | 0 | 0 | 6 of 6 |
| main2-24 | 4 | 52.76 | 0.686 | K3/cost-model x12 | 0 | 0 | 12 of 12 |
| main2-24 | 8 | 79.85 | 1.429 | K0/cost-model x23 K3/cost-model x1 | 0 | 0 | 24 of 24 |
| f1c-2 | 1 | 25.27 | 0.294 | K3/cost-model x3 | 0 | 0 | 3 of 3 |
| f1c-2 | 2 | 36.69 | 0.296 | K3/cost-model x6 | 0 | 0 | 6 of 6 |
| f1c-2 | 4 | 54.08 | 0.892 | K3/cost-model x12 | 0 | 0 | 12 of 12 |
| f1c-2 | 8 | 81.2 | 1.432 | K0/cost-model x23 K3/cost-model x1 | 0 | 0 | 24 of 24 |
| f1c-1 | 1 | 25.39 | 0.292 | K3/cost-model x3 | 0 | 0 | 3 of 3 |
| f1c-1 | 2 | 36.85 | 0.295 | K3/cost-model x6 | 0 | 0 | 6 of 6 |
| f1c-1 | 4 | 53.86 | 0.914 | K3/cost-model x12 | 0 | 0 | 12 of 12 |
| f1c-1 | 8 | 80.95 | 1.242 | K0/cost-model x22 K3/cost-model x2 | 0 | 0 | 24 of 24 |
| f1c-3 | 1 | 25.41 | 0.292 | K3/cost-model x3 | 0 | 0 | 3 of 3 |
| f1c-3 | 2 | 36.75 | 0.295 | K3/cost-model x6 | 0 | 0 | 6 of 6 |
| f1c-3 | 4 | 53.99 | 0.89 | K3/cost-model x12 | 0 | 0 | 12 of 12 |
| f1c-3 | 8 | 81.66 | 1.447 | K0/cost-model x23 K3/cost-model x1 | 0 | 0 | 24 of 24 |
| f1d-1 | 1 | 25.0 | 0.296 | K3/cost-model x3 | 0 | 0 | 3 of 3 |
| f1d-1 | 2 | 36.4 | 0.297 | K3/cost-model x6 | 0 | 0 | 6 of 6 |
| f1d-1 | 4 | 54.05 | 2.225 | K3/cost-model x12 | 0 | 0 | 12 of 12 |
| f1d-1 | 8 | 81.0 | 1.215 | K0/cost-model x22 K3/cost-model x2 | 0 | 0 | 24 of 24 |
| f1d-2 | 1 | 25.05 | 0.294 | K3/cost-model x3 | 0 | 0 | 3 of 3 |
| f1d-2 | 2 | 36.4 | 0.298 | K3/cost-model x6 | 0 | 0 | 6 of 6 |
| f1d-2 | 4 | 53.8 | 0.897 | K3/cost-model x12 | 0 | 0 | 12 of 12 |
| f1d-2 | 8 | 80.88 | 1.443 | K0/cost-model x23 K3/cost-model x1 | 0 | 0 | 24 of 24 |
| f1d-3 | 1 | 24.98 | 0.295 | K3/cost-model x3 | 0 | 0 | 3 of 3 |
| f1d-3 | 2 | 36.35 | 0.298 | K3/cost-model x6 | 0 | 0 | 6 of 6 |
| f1d-3 | 4 | 53.72 | 0.899 | K3/cost-model x12 | 0 | 0 | 12 of 12 |
| f1d-3 | 8 | 80.7 | 1.446 | K0/cost-model x23 K3/cost-model x1 | 0 | 0 | 24 of 24 |

0 marks and 0 stalled verdicts in every phase. The draft length at admission equals main's at
c = 1, 2, 4 (K=3 x3, x6, x12). At c = 8 the ramp admits one or two requests at K=3 before the
far rule takes over; that count varies between main's own boots the same way (main2-21 has a
resume-floor admission, the others one K=3). The c = 4 TTFT p50 of f1d-1 (2.2 s) is the 2+2
phase start of memra#1227, seen on main too.

Aggregates equal main's at c = 1, 4 and 8 (24.7 to 25.4, 52.8 to 54.1 and 79.8 to 81.7 tok/s).
**c = 2 has two start modes on every arm**: the two clients' requests run in lockstep (40 tok/s,
TTFT p50 0.46 s) or one tick apart (36 tok/s, TTFT p50 0.30 s), set by whether the first two
requests are admitted in one tick. Over every census boot of this lane and of memra#1223's
(`tools/tables1211.py c2`): the arms without a mark (main, main with memra#1228) started in
lockstep in 5 of 11 boots, the arms with one in 2 of 12, and all six f1c and f1d boots ran one
tick apart. No mark, verdict or admission differs in those phases. The interleaved cell under
"The shipped head" counts the modes per arm.

## The shipped head (`gb10/sweep-head/`, `gb10/long-head/`, `gb10/recovery-head/`, `gb10/normal-c2/`)

The rows above were measured before the branch was rebased on main 98337756cf, which adds
memra#1186 (a sampled burst now carries its pending token, so a sampled session can yield
silently like a greedy one), memra#1209 (a pooled cohort member closes at its own round) and
memra#1217. The independent review then changed three things (see the PR): the refresh drops a
session's count once its mark is gone and also runs for a plain session that rides a pooled
cohort's step, the demotion check rewrites the count every tick, and the late wave re-read
marks only when its verdict runs. `head` is that tree; `main7` is its base. Every bar item was
measured again on it, both arms on one unit (the other Spark unit than the same-unit rows
above).

The sweep, one boot per arm:

| cell | long tokens main / form | short p50 s main / form | short p95 s main / form | marks form | plain/all form | items |
|---|---|---|---|---|---|---|
| 1:0 | 237 / 238 | 0.304 / 0.304 | 0.307 / 0.306 | 0 | 0/99 | ok |
| 2:0 | 77 / 195 | 0.419 / 0.296 | 0.512 / 0.307 | 1 | 190/198 | ok |
| 4:0 | 59 / 155 | 0.823 / 0.353 | 0.925 / 0.674 | 1 | 300/316 | ok |
| 1:200 | 453 / 450 | 0.298 / 0.296 | 0.304 / 0.299 | 0 | 0/61 | 1 WORSE |
| 2:200 | 141 / 226 | 0.312 / 0.323 | 0.32 / 0.327 | 2 | 110/116 | 3 p50 WORSE, 3 p95 WORSE |
| 4:200 | 52 / 189 | 0.717 / 0.388 | 0.731 / 0.615 | 2 | 184/197 | ok |
| 1:1000 | 641 / 643 | 0.272 / 0.266 | 0.28 / 0.276 | 0 | 0/24 | ok |
| 2:1000 | 513 / 513 | 0.288 / 0.281 | 0.294 / 0.293 | 0 | 0/48 | ok |
| 4:1000 | 238 / 238 | 0.219 / 0.215 | 0.405 / 0.394 | 0 | 0/99 | ok |
| 2:0 t1 | 76 / 196 | 0.422 / 0.296 | 0.511 / 0.306 | 1 | 190/198 | ok |
| 4:0 t1 | 49 / 156 | 0.823 / 0.354 | 0.921 / 0.691 | 1 | 300/316 | ok |

`bar1211.py`: items to explain: 1:200: 1 WORSE; 2:200: 3 p50 WORSE; 2:200: 3 p95 WORSE

- The pattern of the hold form's sweep holds on the shipped tree: 2:0 and 4:0 and both sampled cells pass every item (long 155 to 196 tokens against 49 to 77, short p50 0.30 to 0.35 s against 0.42 to 0.82, 95% of newcomers plain).
- 4:200 passes: long 189 against 52, short p50 0.388 s against 0.717, p95 0.615 against 0.731, 184 of 197 plain.
- **2:200 stays the measured trade**: long 226 against 141 (+60%), short p50 0.323 s against 0.312 (+11 ms), p95 0.327 against 0.320 (+7 ms), 110 of 116 plain. The ticks that primed a short (110) spend 112.8 ms in prefill, 76.1 ms in the long stream's decode step and 3.98 ms in the formation wait (`tools/tickparts.py`, cell 4 of `server-head-62.log`).
- 0 marks and the same draft length at every admission in the 1-client and 1000 ms cells. The one flag there, 1:200 long tokens 450 against 453, is a single boot per arm with no mark involved.
- The longest gap of the long stream equals main's in every marked cell (0.92 and 1.76 s at 2:0 and 4:0): the bound still covers the entries before the mark.
- Every long text is prefix-equal to main7's, greedy and sampled. With memra#1186 in both arms the sampled main rows are 76 and 49 tokens (62 to 64 and 35 to 36 before it).

The 120 s cells:

| arm | cell | long tokens per 30 s segment | total | short p50 s per segment | short p50 / p95 s, whole | newcomers plain / all | marks | refreshes |
|---|---|---|---:|---|---|---|---:|---:|
| main7 | 2:0 | 77 / 81 / 76 / 80 | 314 | 0.419 / 0.505 / 0.423 / 0.504 | 0.423 / 0.512 | 0 / 523 | 0 | 0 |
| main7 | 4:0 | 56 / 169 / 169 / 169 | 565 | 0.821 / 0.355 / 0.355 / 0.355 | 0.355 / 0.864 | 1060 / 1180 | 0 | 0 |
| head | 2:0 | 195 / 202 / 200 / 202 | 801 | 0.297 / 0.298 / 0.298 / 0.298 | 0.298 / 0.3 | 794 / 802 | 1 | 396 |
| head | 4:0 | 153 / 169 / 169 / 168 | 661 | 0.355 / 0.356 / 0.356 / 0.356 | 0.356 / 0.359 | 1312 / 1328 | 1 | 327 |

2:0 holds in all four segments (195 / 202 / 200 / 202 against 77 / 81 / 76 / 80; 396 refreshes, one mark, 794 of 802 newcomers plain). 4:0: the first segment (153 against 56) and the whole-cell short p95 (0.359 s against 0.864); segments 2 to 4 tie main, which recovers by itself there. Long texts prefix-equal.

Recovery with the long stream open:

| arm | pressure cell | follow starts after s | window marks | window newcomers plain / all | long open through the follow | follow K at admission | follow refreshes | follow agg tok/s | long tokens in the follow | follow texts equal across arms |
|---|---|---:|---:|---|---|---|---:|---:|---:|---|
| grace2 | 2:0 | 5.0 | 0 | 0 / 131 | yes | K3/cost-model x6 | - | 29.98 | 595 | 6 of 6 |
| grace2 | 2:0 | 70.0 | 0 | 0 / 131 | yes | K3/cost-model x6 | - | 31.03 | 588 | 6 of 6 |
| grace2 | 2:200 | 70.0 | 0 | 0 / 118 | True / False | K3/cost-model x6 | - | 31.73 | 538 | 6 of 6 |
| f1c | 2:0 | 5.0 | 1 | 192 / 200 | yes | K0/cost-model x4 K3/cost-model x2 | - | 29.04 | 391 | 6 of 6 |
| f1c | 2:0 | 70.0 | 1 | 192 / 200 | yes | K3/cost-model x6 | - | 39.82 | 197 | 6 of 6 |
| f1c | 2:200 | 70.0 | 2 | 110 / 116 | yes | K3/cost-model x6 | - | 39.75 | 197 | 6 of 6 |
| f1d | 2:0 | 5.0 | 1 | 190 / 198 | yes | K0/cost-model x6 | 3 | 25.0 | 486 | 6 of 6 |
| f1d | 2:0 | 70.0 | 1 | 188 / 196 | yes | K3/cost-model x6 | 0 | 38.94 | 197 | 6 of 6 |
| f1d | 2:200 | 70.0 | 2 | 110 / 115 | yes | K3/cost-model x6 | 0 | 38.84 | 197 | 6 of 6 |
| head | 2:0 | 5.0 | 1 | 190 / 198 | yes | K0/cost-model x6 | 3 | 25.21 | 486 | 6 of 6 |
| head | 2:0 | 70.0 | 1 | 190 / 198 | yes | K3/cost-model x6 | 0 | 36.44 | 218 | 6 of 6 |
| head | 2:200 | 70.0 | 2 | 109 / 115 | yes | K3/cost-model x6 | 0 | 36.45 | 218 | 6 of 6 |

The `head` rows repeat f1d's: all six follow admissions plain 5 s after the pressure (3 refreshes, 25.21 tok/s against main's 29.98), K=3 x6 after 70 s of quiet with the demoted stream still open. Texts equal across every arm, 6 of 6 per follow. The main arm of this section is grace2 (main with memra#1228, the other unit); the comparison is the draft length at admission, not a rate.

The c = 2 start mode, interleaved (main7, head, head, main7, head, main7, main7, head; each
boot runs c = 1 then c = 2 four times, the census shape; `tools/c2stats.py`):

| boot | phase | agg tok/s | first pair admitted | ended in one burst | marks | first pair's tokens after each burst |
|---|---:|---:|---|---|---:|---|
| main7-81 | 1 | 36.33 | one tick apart | no | 0 | 1 1 35 70 36 103 70 135 102 160 128 |
| main7-81 | 3 | 37.71 | same tick | no | 0 | 1 1 35 33 69 65 104 97 136 129 160 148 |
| main7-81 | 5 | 40.7 | same tick | yes | 0 | 1 1 33 33 66 68 98 102 130 134 160 160 |
| main7-81 | 7 | 40.38 | same tick | yes | 0 | 1 1 34 35 69 67 101 99 133 131 160 160 |
| head-82 | 1 | 37.69 | same tick | no | 0 | 1 1 36 35 70 70 102 103 137 135 160 158 |
| head-82 | 3 | 38.27 | same tick | no | 0 | 1 1 35 33 69 65 104 97 136 129 160 148 |
| head-82 | 5 | 37.29 | one tick apart | no | 0 | 1 1 33 66 33 98 68 130 102 160 134 |
| head-82 | 7 | 40.43 | same tick | yes | 0 | 1 1 34 35 69 67 101 99 133 131 160 160 |
| head-83 | 1 | 37.84 | same tick | no | 0 | 1 1 35 36 70 70 103 102 135 137 158 160 |
| head-83 | 3 | 36.98 | one tick apart | no | 0 | 1 1 35 69 33 104 65 136 97 160 129 |
| head-83 | 5 | 37.28 | one tick apart | no | 0 | 1 1 33 66 33 98 68 130 102 160 134 |
| head-83 | 7 | 37.42 | one tick apart | no | 0 | 1 1 34 69 35 101 67 133 99 160 131 |
| main7-84 | 1 | 37.73 | same tick | no | 0 | 1 1 35 36 70 70 103 102 135 137 158 160 |
| main7-84 | 3 | 38.19 | same tick | no | 0 | 1 1 35 33 69 65 104 97 136 129 160 148 |
| main7-84 | 5 | 40.7 | same tick | yes | 0 | 1 1 33 33 66 68 98 102 130 134 160 160 |
| main7-84 | 7 | 40.44 | same tick | yes | 0 | 1 1 34 35 69 67 101 99 133 131 160 160 |
| head-85 | 1 | 36.66 | one tick apart | no | 0 | 1 1 35 70 36 103 70 135 102 160 128 |
| head-85 | 3 | 38.13 | same tick | no | 0 | 1 1 35 33 69 65 104 97 136 129 160 148 |
| head-85 | 5 | 37.18 | one tick apart | no | 0 | 1 1 33 66 33 98 68 130 102 160 134 |
| head-85 | 7 | 37.25 | one tick apart | no | 0 | 1 1 34 69 35 101 67 133 99 160 131 |
| main7-86 | 1 | 36.67 | one tick apart | no | 0 | 1 1 35 70 36 103 70 135 102 160 128 |
| main7-86 | 3 | 38.11 | same tick | no | 0 | 1 1 33 35 65 69 97 104 129 136 148 160 |
| main7-86 | 5 | 40.61 | same tick | yes | 0 | 1 1 33 33 66 68 98 102 130 134 160 160 |
| main7-86 | 7 | 37.35 | one tick apart | no | 0 | 1 1 34 69 35 101 67 133 99 160 131 |
| main7-87 | 1 | 36.65 | one tick apart | no | 0 | 1 1 35 70 36 103 70 135 102 160 128 |
| main7-87 | 3 | 38.13 | same tick | no | 0 | 1 1 35 33 69 65 104 97 136 129 160 148 |
| main7-87 | 5 | 40.66 | same tick | yes | 0 | 1 1 33 33 66 68 98 102 130 134 160 160 |
| main7-87 | 7 | 37.29 | one tick apart | no | 0 | 1 1 34 69 35 101 67 133 99 160 131 |
| head-88 | 1 | 37.7 | same tick | no | 0 | 1 1 36 35 70 70 102 103 137 135 160 158 |
| head-88 | 3 | 36.93 | one tick apart | no | 0 | 1 1 35 69 33 104 65 136 97 160 129 |
| head-88 | 5 | 37.21 | one tick apart | no | 0 | 1 1 33 66 33 98 68 130 102 160 134 |
| head-88 | 7 | 40.34 | same tick | yes | 0 | 1 1 35 34 67 69 99 101 131 133 160 160 |

- main7: 11 of 16 c = 2 phases started in the same tick; mean aggregate 38.60 tok/s; marks 0
- head: 7 of 16 c = 2 phases started in the same tick; mean aggregate 37.79 tok/s; marks 0
- Fisher exact, two-sided, same-tick starts 11/16 against 7/16: p = 0.285

| start mode | main7: phases, agg tok/s mean (range) | head: phases, agg tok/s mean (range) |
|---|---|---|
| same tick, ended in one burst | 6, 40.58 (40.38-40.7) | 2, 40.39 (40.34-40.43) |
| same tick, ended apart | 5, 37.97 (37.71-38.19) | 5, 37.93 (37.69-38.27) |
| one tick apart | 5, 36.86 (36.33-37.35) | 9, 37.13 (36.66-37.42) |

The start modes are set by a race between the second client's first request and the worker's
admission pass: both requests admitted in one tick and ending in one burst (40.4 to 40.7 tok/s),
admitted in one tick and ending apart (37.7 to 38.3), or admitted one tick apart (36.3 to 37.4).
Interleaved on one unit, 16 phases per arm: main started in the same tick in 11, the shipped
tree in 7 (Fisher exact, two-sided, p = 0.29). By phase position (first to fourth c = 2 phase of
a boot) the same-tick counts go both ways: 1 of 4 against 3 of 4, 4 against 2, 4 against 0, 2
against 2. The aggregate per mode is equal across arms (40.58 against 40.39, 37.97 against
37.93, 36.86 against 37.13 tok/s), so the arms differ only in how often each mode occurs; mean
aggregate 38.60 against 37.79 tok/s (-2.1%), and the bound if the slowest mode replaced the
fastest on every start is -8.5%. No phase sets a mark, and no verdict or admission differs. No
code path was found that would move the race (the change adds microseconds to an admission,
which would favour the same tick). Pooled with the earlier traced census boots (not interleaved:
5 of 8 against 0 of 6) the count is 16 of 24 against 7 of 22, p = 0.04, so a shift is not
excluded; it is tracked in memra#1240.

## Exactness

- **Seeded-invariance** (`tools/seeded-invariance-gate.py --penalised device`), GB10: f1d
  (`gb10/inv/`) PASS, 54 rows, 10 prompts, cases ref, K = 0..3 and demote; `head`
  (`gb10/inv-head/`) PASS, 54 rows. f1c on the RTX 5090 (`rtx5090/inv-f1c/`): PASS, 54 rows.
- **Burst-boundary** (`tools/spec-burst-boundary-gate.py`), GB10: f1d with `--carry no`, a
  binary from before memra#1186 (`gb10/bnd/`): PASS, 229 rows; `head` (`gb10/bnd-head/`)
  PASS, 229 rows (1,252 sampled bursts entered on a carried token, none drew at a burst's tail).
- **Stall cells**: every long text prefix-equal to the main arm's, greedy and sampled, in every
  sweep directory.

## RTX 5090 Laptop GPU (`rtx5090/`)

Measured on f1c against `grace`. The refresh acts only on a live mark, and this card sets none.

| model, gate owner | cell | main: long tokens, short p50 s | f1c: long tokens, short p50 s | marks | newcomers |
|---|---|---|---|---:|---|
| 27B, band | 2:0 | 1197, 0.887 | 1197, 0.888 | 0 | K=0 (concurrency) x68 both |
| 27B, band | 2:200 | 1266, 0.866 | 1264, 0.866 | 0 | K=0 x56 both |
| 27B, band | 4:0 | 176, 0.348 | 176, 0.348 | 0 | K=0 x348 both |
| Ornith, cost model | 2:0 | 542, 0.111 | 544, 0.111 | 0 | K=0 (cost model) x542, x544 |
| Ornith, cost model | 2:200 | 2008, 0.104 | 1990, 0.103 | 0 | K=0 x198 both |
| Ornith, cost model | 4:0 | 404, 0.149 | 404, 0.149 | 0 | K=0 x808 both |
| 27B, `MEMRA_SPEC_GATE=0` | 2:0 | - | 72, 0.473 | 0 | K=3 x129 |
| 27B, `MEMRA_SPEC_GATE=0` | 4:0 | - | 36, 0.897 | 0 | K=3 x135 |

- No mark can act where the band owns the gate (no cost entry) or with no gate owner; the
  gate-off rows equal the entry-yield bound's own (72 to 74 and 36 tokens).
- Ornith's cost model already picks plain from n=2, so nothing speculates long enough to press.
- The c = 1..8 census (`bandnormal4/`, `ornnormal4/`): 0 marks, the same admissions on both
  arms except one band admission at c=4 (K=3 cold-short on main, K=1 resume-floor on f1c: the
  band's arrival timing), every text equal.
- Long texts prefix-equal to main's in every stall cell.

## CPU: each test fails on the shape that was rejected

`crates/memra-server/src/worker.rs` tests `spec_cost_stall_mark_does_not_spread`,
`spec_cost_stall_mark_expires`, `spec_cost_stall_hold_ends_with_its_mark`,
`spec_cost_stall_refresh_keeps_only_a_live_mark`, `spec_press_history_marks_at_four`,
`spec_stall_mark_is_wired`. Twelve red arms, each a one-edit variant of the shipped code
(`tools/dd-redgreen.sh`), each failing its target test; green passes all:

| red arm | the variant | fails |
|---|---|---|
| red-slot | the slot probe's storage: three zero-rate samples in the realized slot, no mark read | the spread test (n=2 prices 33 / 42 / 47 become 12 / 15 / 17 tok/s), the expiry test, the hold test |
| red-wiring | the tick's stall sweep removed | the wiring test |
| red-ntick | no admission marks (the tick's count only) | the spread test (`the waiting request marks its wave`) |
| red-nohold | no held admission | the hold test |
| red-freshttl | a held wave on a fresh time-to-live | the hold test |
| red-nodemotekey | the demotion sites without the session's stall count | the wiring test |
| red-refreshcreates | a refresh that creates a mark | the refresh test |
| red-norefreshwire | the tick's refresh removed | the wiring test |
| red-staledrop | a refresh that keeps a count whose mark is gone | the refresh test |
| red-holdmin | a hold on the earliest expiry among the live marks | the hold test |
| red-stickycount | a demotion check that never rewrites the count | the wiring test |
| red-latemark | the late re-read marks even when its verdict does not run | the wiring test |

## Not measured

- The agentic c = 1..8 cell of `research/batch-spec-20261101/GB10-AGENTIC.md` (48 turns, T=1)
  and stall cells at 8 clients. The c = 1..8 census above is the ordinary-traffic proof here.
- The RTX 5090 on `f1d` or `head`: its rows are f1c's. The later commits act only on a live
  mark or on a session a mark demoted, and that card sets no mark.
- Any card other than GB10 where the cost model owns the gate and speculation is ahead at
  n >= 2.

## Files

- `gb10/first-forms/`, `gb10/sweep-exact/`, `gb10/sweep/`, `gb10/sweep-head/`: `stall.jsonl`
  (one row per cell), `boots.log`, `guard.log` (binary hash, driver, load), server logs, long
  texts. Server logs over 60 kB are gzip-compressed (`server-*.log.gz`); `tickparts.py` and `c2stats.py`
  read them as they are.
- `gb10/long-hold/`, `gb10/long/`, `gb10/long-same-unit/`, `gb10/long-head/`,
  `gb10/recovery-hold/`, `gb10/recovery/`, `gb10/recovery-head/`: the same layout.
- `gb10/normal-hold/`, `gb10/normal/`, `gb10/normal-c2/`: `normal.jsonl` and server logs.
- `gb10/inv/`, `gb10/bnd/`, `gb10/inv-head/`, `gb10/bnd-head/`, `rtx5090/inv-f1c/`: gate rows,
  logs and texts.
- `tools/`: `stall_cell.py`, `normal_cell.py`, `bar1211.py`, `tables1211.py`, `c2stats.py`,
  `prefix1211.py`, `tickparts.py`, `dd-redgreen.sh`.
