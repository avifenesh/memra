# A newcomer primes before the pooled cohorts and joins the tick's lockstep (memra#1227)

Date: 2026-11-12. Qwen3.8-27B NVFP4 + Q5_K with its MTP head, context 8,192, eight sessions,
every switch at its default, `MEMRA_TICK_TRACE=1` on the cells. Rate and latency cells: the N of
every figure is in its row; a boot is one server start, one boot per device lease. Decision
record: [`docs/decisions/SPEC-COHORT-NEWCOMER.md`](../../docs/decisions/SPEC-COHORT-NEWCOMER.md).
Follows `research/spec-stall-mark-20261111/` (memra#1211), whose head is the base arm here.

**Verdict: adopted (memra#1227).** An MTP speculative session that has emitted nothing and
has a same-model speculative peer now primes before the tick's pooled cohorts and joins that
tick's lockstep. On GB10, both arms interleaved
on one unit: a newcomer's first token behind two or three long speculating streams comes at
0.27 and 0.29 s where it came at 1.86 and 2.03 s (p50, 24 newcomers per arm and cell); behind
one stream its second token follows the first by 0.005 s where it waited 1.26 s. Closed-loop
clients gain 9 to 11% aggregate tok/s at c = 2, 3 and 4 (four phases per arm) because a
newcomer shares the cohort's forwards one burst earlier. Every text is equal across arms,
greedy and seeded sampled, and the seeded-invariance and burst-boundary gates pass. A newcomer
with a multi-chunk prime (5,117 tokens) behind two streams gets its first token at 3.99 s where
it came at 5.35 s. Measured
costs: a cohort under short arrivals with a think time now meets waiting requests at its burst
entries and takes memra#1211's stall mark (two long streams, two clients at 200 ms think: 43%
fewer long tokens for half the short latency and 65% more short requests); behind ONE long stream that stream's gap per newcomer is one of 0.40 s where it was two
of 0.31 s; at c = 2 closed loop the first-token p50 is 0.40 s where it was 0.30 s (the two
clients now end in the same burst, so their next requests arrive together), with the p95
unchanged and each request ending about 1 s sooner. Not covered: a newcomer admitted to plain
decode (memra#1244).

## Conditions

- **GB10**: one DGX Spark unit per comparison (both arms on that unit), the calibrated cost
  model owns the gate. `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`.
- **RTX 5090 Laptop GPU**: the same 27B; its calibration is refused on this card (memra#1191),
  so the band owns the gate.
- **Binaries** (GB10 `121a`, the 5090 `120a`):

  | arm | tree | binary |
  |---|---|---|
  | `base` | a20fd8c59f, memra#1241's head (crates equal main 340a5e5bfa) | GB10 `memra-0.141.1-5b4e956cdb89`, sha256 `03d6896b4970...` (built on each unit, byte-identical); 5090 sha256 `c9e96631dc03...` |
  | `change` | 301df182c6 on a20fd8c59f: the first form | GB10 `memra-0.141.1-51f0eb72655f`, sha256 `9cbc34a774d4...`; 5090 sha256 `3d08768c1f22...` |
  | `change2` | bab448b40d (on main 68b46bfc18 as 0c8b08e0f9, crates unchanged): the final form, `change` with the review's fixes (only a trimmed prime steps first; the cohort pass reads the prime quantum before the cohorts form) | GB10 `memra-0.141.1-2c65e91e03d3`, sha256 `73cfa6537b19...`; 5090 sha256 `1e800ab9fd85...` |
  | `gb10/repro` only | memra#1241's head binary from that lane, built on the engine tree before its rebase | sha256 `777a202764ec...` |

- **Cells**:
  - `tools/newcomer_cell.py`: L long streamed requests (max_tokens 3000) run alone for 8 s,
    then 12 streamed 16-token newcomers are sent one at a time at a fixed list of offsets after
    the previous one ended; the long streams run 4 s more. A cell is `L:class[:newcomer class]`;
    `t1` is temperature 1 with fixed seeds, `pen` is greedy with a frequency penalty (admitted
    to plain decode).
  - `normal_cell.py` of `research/spec-stall-mark-20261111/tools/`: c closed-loop clients, three
    greedy 160-token streamed requests each.
  - `tools/stall_cell.py`: memra#1241's stall cell with `LONGS=2` long streams.
- **Leases**: one boot per lease, both arms of a comparison on one unit: newcomer cells on unit
  A (order base, change, change, base, and the same order for `change2`), closed loop and
  stall cells on unit B (change, base, base, change; the stall boots base, change, change,
  base, change2, base).
- **Which form ran what**: `change` (the first form) ran the newcomer cell, the closed loop,
  the stall cells' first three boots per arm pair and the RTX 5090 cells. `change2` (the
  final form) ran the long-prompt cell with its short-prompt cells again, run 2 of the 0 ms
  stall cells, the exactness gates and one RTX 5090 boot. For a newcomer with a same-model
  speculative peer and a one-chunk prime, which is every newcomer of the first-form cells, the
  two forms run the same code.

## Where the wait came from (`gb10/repro/`)

One boot of the base on GB10, tick trace on. With two long streams live, a newcomer admitted
at a tick top:

```
[spec-k] model="q" tenant="" K=3 source=cost-model prompt=34 cached=0 lcp=0 active=3 wave=3
[tick-spec] seq=1 slot=2 start_ms=1742.442 wall_ms=217.631 generated=0->1      <- the cohort's burst (1742 ms), then the prime (218 ms)
[tick-spec] seq=2 slot=0 start_ms=1960.087 wall_ms=0.001 generated=344->379    <- the cohort members' results
[tick-spec] seq=3 slot=1 start_ms=1960.101 wall_ms=0.001 generated=305->337
[tick] act=3 ... spec=3 spec_calls=3 spec_ms=1960.1
```

With one long stream there is no cohort, so the newcomer primes first, and its prime burst
ends at one token because a peer is live. The stream's serial burst then runs before the
newcomer's second token:

```
[tick-spec] seq=1 slot=1 start_ms=0.004 wall_ms=212.668 generated=0->1         <- the prime
[tick-spec] seq=2 slot=0 start_ms=212.684 wall_ms=1195.731 generated=368->400  <- the long stream's serial burst
[tick] act=2 ... spec=2 spec_calls=2 spec_ms=1408.4
[tick-spec] seq=2 slot=1 start_ms=1638.674 generated=1->16                     <- next tick: the cohort of two
```

Every newcomer behind a cohort pays it (12 of 12 at L = 2 and at L = 3), not only some starts
of an ordinary cell as memra#1227 recorded: an arrival ends the cohort's burst at its round
(the admission yield), the next tick admits it, and the cohort then runs a whole burst (32
tokens per member) before the prime.

## The newcomer cell, GB10 (`gb10/newcomer/`)

Two boots per arm on one unit, 12 newcomers per cell and boot (24 per row). `summ_newcomer.py`
prints the table and the text check.

| long streams | class (long/new) | arm | boots | newcomers | TTFT p50 s | TTFT p95 s | TTFT max s | 1st to 2nd token p50 s | newcomer E2E p50 s | long tok/s (sum), per boot | all tok/s (long + newcomers), per boot | long longest gap s, max | long ITL p50 / p99 ms |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| 0 | greedy/greedy | base | 2 | 24 | 0.211 | 0.221 | 0.254 | 0.099 | 0.709 | - | 7.73, 7.76 | - | - |
| 0 | greedy/greedy | change | 2 | 24 | 0.210 | 0.225 | 0.252 | 0.098 | 0.703 | - | 7.77, 7.76 | - | - |
| 1 | greedy/greedy | base | 2 | 24 | 0.267 | 0.302 | 0.302 | 1.262 | 2.184 | 23.07, 23.31 | 27.67, 27.95 | 0.316 | 0.0 / 309.8 |
| 1 | greedy/greedy | change | 2 | 24 | 0.273 | 0.300 | 0.300 | 0.005 | 0.912 | 21.67, 21.7 | 28.85, 28.89 | 0.403 | 0.0 / 396.2 |
| 1 | greedy/pen | base | 2 | 24 | 0.125 | 1.761 | 1.783 | 0.074 | 1.289 | 21.22, 21.42 | 22.68, 22.89 | 18.691 | 0.0 / 174.6 |
| 1 | greedy/pen | change | 2 | 24 | 0.125 | 1.730 | 1.737 | 0.074 | 1.272 | 21.52, 21.46 | 23.0, 22.94 | 18.561 | 0.0 / 172.4 |
| 2 | greedy/greedy | base | 2 | 24 | 1.863 | 2.152 | 2.173 | 0.007 | 2.563 | 39.26, 39.85 | 43.31, 43.92 | 0.911 | 0.0 / 415.0 |
| 2 | greedy/greedy | change | 2 | 24 | 0.268 | 0.308 | 0.319 | 0.007 | 1.015 | 36.98, 36.82 | 43.81, 43.65 | 0.607 | 0.0 / 414.6 |
| 2 | greedy/pen | base | 2 | 24 | 0.132 | 1.770 | 1.851 | 0.075 | 1.296 | 36.7, 36.73 | 37.94, 37.98 | 18.554 | 0.0 / 287.3 |
| 2 | greedy/pen | change | 2 | 24 | 0.125 | 1.750 | 1.758 | 0.074 | 1.273 | 37.11, 36.97 | 38.37, 38.23 | 18.421 | 0.0 / 283.5 |
| 2 | t1/t1 | base | 2 | 24 | 1.881 | 2.159 | 2.214 | 0.007 | 2.704 | 36.95, 36.95 | 40.95, 40.96 | 0.993 | 0.0 / 514.4 |
| 2 | t1/t1 | change | 2 | 24 | 0.267 | 0.316 | 0.317 | 0.007 | 1.093 | 36.25, 36.27 | 42.96, 43.01 | 0.702 | 0.0 / 414.3 |
| 3 | greedy/greedy | base | 2 | 24 | 2.029 | 2.412 | 2.419 | 0.009 | 2.934 | 51.41, 51.37 | 55.18, 55.14 | 0.962 | 0.0 / 569.6 |
| 3 | greedy/greedy | change | 2 | 24 | 0.289 | 0.336 | 0.336 | 0.009 | 1.144 | 48.32, 47.83 | 54.76, 54.26 | 0.926 | 0.0 / 449.1 |

`prefix check: 44 long texts, 336 newcomer texts, 0 mismatches -> PASS`

- **First token behind a cohort**: 1.863 s to 0.268 s behind two streams, 2.029 s to 0.289 s
  behind three, 1.881 s to 0.267 s for seeded sampled requests (T = 1). The worst newcomer of
  the change (0.34 s) is ahead of the best of the base behind a cohort (1.5 s).
- **Second token behind one stream**: 1.262 s to 0.005 s, and the newcomer's 16 tokens end at
  0.91 s where they ended at 2.18 s.
- **"All tok/s"** is the long streams' tokens inside the newcomer window plus the newcomers'
  own, over the window. The windows differ in length across arms (a newcomer that waits less
  ends sooner, and the schedule is relative to the previous newcomer's end), so the long
  streams' rate alone does not compare: it is about 6% lower on the change in every cohort
  cell (23.2 to 21.7, 39.6 to 36.9 and 51.4 to 48.1 tok/s behind one, two and three streams),
  because the same twelve newcomers, each a prime and fifteen rounds in a wider cohort, fall in
  a window 36 to 41% shorter. All tok/s: +4% behind one stream, +0.3% behind two,
  -1.2% behind three (54.51 against 55.16 tok/s, mean of two boots).
- **Each member's gap per newcomer** (the long streams' gaps above 150 ms inside the window,
  one stream per row of `long_streams`): behind one stream the base has two gaps of 0.31 s per
  newcomer (its prime, then the two-member tick), the change one of 0.40 s (the prime and a
  two-member round together). Behind two streams the median of each member's twelve longest
  gaps is 0.41 to 0.51 s on the base and 0.41 s on the change, behind three 0.44 to 0.64 s and
  0.44 s; the longest gap falls or holds in every cohort cell (0.91 to 0.61 s, 0.99 to 0.70 s,
  0.96 to 0.93 s).
- **A plain newcomer** (`pen` rows) is equal across arms and is memra#1244: while the long
  streams live it gets one token per tick, 1.4 to 1.7 s apart (16 tokens in 21 to 28 s behind
  two streams; 1.29 s once they ended at 3,000 tokens, which is the 18 s "gap" of those rows).
- **Texts**: 44 long texts prefix-equal and 336 newcomer texts equal across the four boots.

## Long prompts and the final form, GB10 (`gb10/newcomer-long/`)

The review's case: a newcomer whose prime spans chunks. The same cell with 4,200-word newcomer
prompts (5,117 tokens; rows `+long`) behind one and two long streams, and the short-prompt
cells again, on `change2`. Two boots per arm on one unit (base, change2, change2, base), 24
newcomers per row.

| long streams | class (long/new) | arm | boots | newcomers | TTFT p50 s | TTFT p95 s | TTFT max s | 1st to 2nd token p50 s | newcomer E2E p50 s | long tok/s (sum), per boot | all tok/s (long + newcomers), per boot | long longest gap s, max | long ITL p50 / p99 ms |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| 1 | greedy/greedy | base | 2 | 24 | 0.284 | 0.307 | 0.308 | 1.189 | 2.229 | 22.8, 22.8 | 27.33, 27.32 | 0.317 | 0.0 / 313.6 |
| 1 | greedy/greedy | change2 | 2 | 24 | 0.285 | 0.305 | 0.308 | 0.005 | 0.939 | 21.16, 21.07 | 28.2, 28.09 | 0.418 | 0.0 / 401.7 |
| 1 | greedy/greedy+long | base | 2 | 24 | 3.917 | 3.945 | 3.951 | 0.006 | 4.407 | 7.88, 7.89 | 10.07, 10.08 | 2.522 | 0.0 / 2511.6 |
| 1 | greedy/greedy+long | change2 | 2 | 24 | 3.928 | 4.044 | 4.047 | 0.006 | 4.449 | 9.21, 9.15 | 11.37, 11.31 | 2.599 | 0.0 / 2518.9 |
| 2 | greedy/greedy | base | 2 | 24 | 1.881 | 2.042 | 2.045 | 0.007 | 2.616 | 38.61, 38.64 | 42.68, 42.72 | 1.2 | 0.0 / 416.8 |
| 2 | greedy/greedy | change2 | 2 | 24 | 0.266 | 0.316 | 0.317 | 0.007 | 0.988 | 36.31, 36.55 | 43.17, 43.44 | 0.617 | 0.0 / 420.1 |
| 2 | greedy/greedy+long | base | 2 | 24 | 5.354 | 5.773 | 5.776 | 0.008 | 5.836 | 20.09, 20.08 | 21.88, 21.88 | 3.109 | 0.0 / 2513.7 |
| 2 | greedy/greedy+long | change2 | 2 | 24 | 3.987 | 4.056 | 4.066 | 0.008 | 4.494 | 15.81, 15.84 | 18.01, 18.04 | 2.608 | 0.0 / 2524.4 |
| 3 | greedy/greedy | base | 2 | 24 | 2.027 | 2.159 | 2.397 | 0.009 | 2.857 | 51.08, 51.09 | 54.88, 54.92 | 1.026 | 0.0 / 534.7 |
| 3 | greedy/greedy | change2 | 2 | 24 | 0.303 | 0.338 | 0.339 | 0.009 | 1.130 | 47.97, 47.77 | 54.46, 54.24 | 0.652 | 0.0 / 447.0 |

`prefix check: 36 long texts, 240 newcomer texts, 0 mismatches -> PASS`

- **A long prompt behind a cohort**: first token at 3.99 s where it came at 5.35 s (p50; max
  4.07 against 5.78 s). On the base the cohort ran a whole burst before the newcomer's first
  chunk; on `change2` the first chunk runs first and the cohort runs one round between
  chunks. Behind one stream nothing moves (3.92 and 3.93 s): there was no cohort burst ahead
  of the prime, and the first token of a long prompt is its prime.
- **What the cohort pays.** A member's gap per long newcomer is one prime chunk on both arms
  (2.5 s, the median of its twelve longest gaps; longest 3.1 s on the base, 2.6 s on
  `change2`). The long streams emit 1,105 tokens in the newcomer window where they emitted
  1,722 (two boots each), and the window is 69.8 s where it was 85.7 s: about 51 tokens per
  newcomer, the burst the cohort used to run ahead of the newcomer's first chunk, in 1.3 s
  less wall. The window's rate (15.8 against 20.1 tok/s for the long streams) reflects that
  the same twelve primes fill more of a shorter window.
- **Short prompts on the final form** repeat the first form's rows: first token behind two
  and three streams 0.266 and 0.303 s against 1.881 and 2.027 s; second token behind one
  stream 0.005 against 1.189 s; all tok/s +3.0%, +1.4%, -1.0%.

## Closed loop, GB10 (`gb10/closed-loop/`)

c = 2, 3, 4, each twice per boot, two boots per arm on one unit (four phases per row).
`summ_norm.py` prints the table and the text check.

| c | arm | phases | aggregate tok/s per phase | mean | TTFT p50 s | TTFT p95 s | TTFT max s | inter-token p99 ms | admissions by K | stall marks | demotions |
|---:|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|
| 2 | base | 4 | 36.55, 36.09, 36.61, 36.14 | 36.35 | 0.301 | 0.483 | 0.490 | 308.5 | K3/cost-model x24 | 0 | 0 |
| 2 | change | 4 | 40.3, 40.68, 40.3, 40.7 | 40.49 | 0.401 | 0.486 | 0.487 | 231.6 | K3/cost-model x24 | 0 | 0 |
| 3 | base | 4 | 45.45, 44.77, 45.56, 44.81 | 45.15 | 0.825 | 2.445 | 2.455 | 438.0 | K3/cost-model x36 | 0 | 0 |
| 3 | change | 4 | 49.0, 49.87, 49.01, 48.58 | 49.11 | 0.452 | 0.835 | 0.837 | 441.4 | K3/cost-model x36 | 0 | 0 |
| 4 | base | 4 | 52.74, 53.38, 53.92, 53.71 | 53.44 | 1.924 | 2.706 | 3.022 | 463.0 | K3/cost-model x48 | 0 | 0 |
| 4 | change | 4 | 58.02, 58.15, 58.1, 58.22 | 58.12 | 0.604 | 0.913 | 1.066 | 459.3 | K3/cost-model x48 | 0 | 0 |

`text check: 216 requests, 54 (phase, prompt) groups, 0 with more than one text, 0 request errors -> PASS`

- Aggregate: +11.4% at c = 2, +8.8% at c = 3, +8.8% at c = 4. A request that starts while
  others run now joins their lockstep in the tick it primes in, where it used to wait a burst
  outside it.
- First token: p50 0.83 to 0.45 s at c = 3 and 1.92 to 0.60 s at c = 4; p95 2.45 to 0.84 s
  and 2.71 to 0.91 s.
- **c = 2 is a measured trade on the first token**: p50 0.30 to 0.40 s, p95 equal (0.48 s),
  and each request ends about 1 s sooner (7.5 to 7.9 s against 8.5 to 9.0 s). On the change
  the two clients run in lockstep from their first tick and end in the same burst, so their
  next requests arrive together and the second waits the first's prime and tick. On the base
  they ran one tick apart in all four phases (the slow start mode of memra#1240's census).
- 0 stall marks, 0 demotions, the same draft length at every admission (K = 3) on both arms.
  216 requests, every text equal.

## The stall cells with two long streams, GB10 (`gb10/stall-2long/`)

memra#1241's cells (closed-loop clients of one-token requests for 30 s, `clients:think ms`)
with two long streams instead of one, so the long streams are a cohort. One unit, two boots per
arm for every cell, arms interleaved (rows `base`, `change`, `change-r2`, `base-r2`,
`change2-r2`). A short request is one token, so all tokens = long tokens + short requests.
`summ_stall.py` prints the per-stream table and the text check.

| cell | arm, run | long tokens in 30 s (both streams) | short requests served | all tokens | short p50 s | short p95 s | stall marks | newcomers admitted plain |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 2:0 | base, run 1 | 361 | 145 | 506 | 0.386 | 0.414 | 2 | 139 of 145 |
| 2:0 | base, run 2 | 364 | 144 | 508 | 0.384 | 0.418 | 2 | 138 of 144 |
| 2:0 | change, run 1 | 381 | 192 | 573 | 0.304 | 0.313 | 1 | 184 of 192 |
| 2:0 | change, run 2 (final form) | 382 | 194 | 576 | 0.301 | 0.313 | 1 | 186 of 194 |
| 4:0 | base, run 1 | 295 | 233 | 528 | 0.456 | 0.935 | 2 | 216 of 233 |
| 4:0 | base, run 2 | 296 | 235 | 531 | 0.453 | 0.933 | 2 | 218 of 235 |
| 4:0 | change, run 1 | 305 | 308 | 613 | 0.362 | 0.755 | 1 | 292 of 308 |
| 4:0 | change, run 2 (final form) | 306 | 312 | 618 | 0.358 | 0.748 | 1 | 296 of 312 |
| 2:200 | base, run 1 | 757 | 67 | 824 | 0.664 | 0.672 | 0 | 0 of 67 |
| 2:200 | base, run 2 | 770 | 68 | 838 | 0.655 | 0.665 | 0 | 0 of 68 |
| 2:200 | change, run 1 | 435 | 111 | 546 | 0.336 | 0.352 | 2 | 106 of 111 |
| 2:200 | change, run 2 | 437 | 111 | 548 | 0.335 | 0.349 | 2 | 106 of 111 |
| 4:200 | base, run 1 | 462 | 104 | 566 | 0.855 | 1.094 | 0 | 0 of 104 |
| 4:200 | base, run 2 | 330 | 181 | 511 | 0.409 | 0.735 | 2 | 168 of 181 |
| 4:200 | change, run 1 | 371 | 193 | 564 | 0.401 | 0.633 | 2 | 180 of 193 |
| 4:200 | change, run 2 | 370 | 193 | 563 | 0.397 | 0.640 | 2 | 180 of 193 |

- **Runs.** Two boots per arm for every cell, arms interleaved on one unit. The 200 ms cells
  and run 1 of the 0 ms cells ran the first form of the change (301df182c6); run 2 of the 0 ms
  cells ran the final form (bab448b40d crates). The two forms differ only for a newcomer
  without a same-model speculative peer and for a prime that spans chunks.
- **0 ms think: both classes gain, in both runs.** The long cohort gets 381 and 382 tokens
  where it got 361 and 364 at two clients, and 305 and 306 where it got 295 and 296 at four;
  the short p50 falls from 0.39 to 0.30 s and from 0.45 to 0.36 s, and a third more short
  requests are served. Both arms set a stall mark and demote the streams.
- **2:200 is a stable trade.** The long cohort gets 43% fewer tokens (435 and 437 against 757
  and 770) and the cell 34% fewer tokens in all; the short requests get half the latency (p50
  0.34 s against 0.66 s) and 65% more of them are served (111 against 67 and 68). The base set
  no stall mark in either run; the change set it in both.
- **4:200, n = 2 per arm.** The change set the mark in both runs: 371 and 370 long tokens, 193
  short requests at a p50 of 0.40 s, 564 and 563 tokens in all. The base set the mark in one
  of its two runs: without it 462 long tokens, 104 short requests at 0.86 s, 566 tokens in
  all; with it 330 long tokens, 181 short requests at 0.41 s, 511 tokens in all. Two boots do
  not say how often the base marks.
- **Why.** On the base a one-token request admitted beside the cohort waited out the cohort's
  burst and primed after it, so the next arrival was rarely waiting when a burst began: no
  pressed entries, no stall mark (memra#1211), and the streams kept speculating while each
  short answer took 0.66 to 0.86 s. With the change the short requests prime first, the next
  arrival is waiting when the cohort's burst begins, the mark fires after four such entries,
  the streams demote and newcomers are admitted plain. That is memra#1241's outcome for one
  long stream at 2:200 (226 tokens for the stream, short p50 0.32 s): here 218 and 217 tokens
  per stream at 0.34 s. The share policy memra#1241 landed now reaches a cohort; a cohort no
  longer keeps its rate by making arrivals wait a burst.
- Long texts are prefix-equal across arms and runs in every cell (32 texts).

## Exactness

- **Seeded-invariance** (`tools/seeded-invariance-gate.py --penalised device`), final form:
  GB10 (`gb10/inv/`) PASS, 54 rows, 10 prompts, cases ref, K = 0..3 and demote; RTX 5090 (`rtx5090/inv/`) PASS, 54 rows. First form: GB10
  (`gb10/inv-first-form/`) PASS, 54 rows, 10 prompts, cases ref, K = 0..3 and demote; RTX 5090
  (`rtx5090/inv-first-form/`) PASS, 54 rows.
- **Burst-boundary** (`tools/spec-burst-boundary-gate.py`), final form: GB10 (`gb10/bnd/`)
  PASS, 229 rows, grid default, 1:2, 3:2, 3:3 (1,252 sampled bursts entered on a carried token, none drew at
  a burst's tail). First form: GB10 (`gb10/bnd-first-form/`) PASS, 229 rows, grid default, 1:2,
  3:2, 3:3 (1,252 sampled bursts entered on a carried token, none drew at a burst's tail). An
  earlier first-form run on the other unit lost its ssh connection at 224 of 229 rows, every
  row passing until then, and was stopped; it is not a result and is not kept.
- **Cells**: every long text prefix-equal and every newcomer and closed-loop text equal across
  arms on both devices (the tables' text checks).

## RTX 5090 Laptop GPU (`rtx5090/`)

The no-regression check: the newcomer cell (first form: two boots per arm in the order base,
change, change, base; final form: one boot, then a pair base, change2), the closed-loop cell
c = 1, 2, 4, 8 on the first form (one boot per arm, single runs), seeded-invariance on both
forms.

| long streams | class (long/new) | arm | boots | newcomers | TTFT p50 s | TTFT p95 s | TTFT max s | 1st to 2nd token p50 s | newcomer E2E p50 s | long tok/s (sum), per boot | all tok/s (long + newcomers), per boot | long longest gap s, max | long ITL p50 / p99 ms |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| 0 | greedy/greedy | base | 3 | 36 | 0.140 | 0.152 | 0.162 | 0.035 | 0.318 | - | 9.68, 9.68, 9.67 | - | - |
| 0 | greedy/greedy | change | 2 | 24 | 0.140 | 0.142 | 0.147 | 0.035 | 0.318 | - | 9.68, 9.68 | - | - |
| 0 | greedy/greedy | change2 | 2 | 24 | 0.140 | 0.153 | 0.162 | 0.035 | 0.320 | - | 9.68, 9.67 | - | - |
| 1 | greedy/greedy | base | 3 | 36 | 0.158 | 0.189 | 0.194 | 0.472 | 0.936 | 65.62, 62.66, 65.35 | 72.77, 69.75, 72.47 | 0.194 | 0.0 / 79.1 |
| 1 | greedy/greedy | change | 2 | 24 | 0.157 | 0.175 | 0.175 | 0.002 | 0.438 | 63.22, 63.1 | 72.29, 72.16 | 0.218 | 0.0 / 52.1 |
| 1 | greedy/greedy | change2 | 2 | 24 | 0.155 | 0.172 | 0.175 | 0.002 | 0.438 | 63.22, 62.79 | 72.29, 71.88 | 0.225 | 0.0 / 52.9 |
| 2 | greedy/greedy | base | 3 | 36 | 0.824 | 0.993 | 1.015 | 0.035 | 2.008 | 82.33, 78.76, 80.5 | 87.25, 83.56, 85.27 | 0.479 | 0.0 / 198.8 |
| 2 | greedy/greedy | change | 2 | 24 | 0.834 | 0.922 | 0.924 | 0.034 | 1.984 | 83.35, 83.42 | 88.23, 88.3 | 0.386 | 0.0 / 155.5 |
| 2 | greedy/greedy | change2 | 2 | 24 | 0.859 | 1.003 | 1.010 | 0.035 | 2.056 | 82.06, 75.48 | 86.9, 80.15 | 0.418 | 0.0 / 205.5 |
| 2 | t1/t1 | base | 3 | 36 | 0.944 | 1.051 | 1.073 | 0.035 | 2.136 | 71.23, 70.45, 68.15 | 75.88, 75.1, 72.75 | 0.474 | 0.0 / 208.6 |
| 2 | t1/t1 | change | 2 | 24 | 0.919 | 1.007 | 1.010 | 0.035 | 2.092 | 71.86, 71.0 | 76.58, 75.72 | 0.385 | 0.0 / 198.9 |
| 2 | t1/t1 | change2 | 2 | 24 | 0.974 | 1.097 | 1.132 | 0.035 | 2.211 | 68.1, 68.31 | 72.66, 72.83 | 0.426 | 0.0 / 209.6 |
| 3 | greedy/greedy | base | 3 | 36 | 0.143 | 0.205 | 0.207 | 0.031 | 0.646 | 95.66, 93.4, 94.5 | 103.83, 101.47, 102.63 | 0.204 | 28.6 / 157.3 |
| 3 | greedy/greedy | change | 2 | 24 | 0.137 | 0.149 | 0.151 | 0.031 | 0.632 | 97.99, 97.72 | 106.2, 105.95 | 0.157 | 28.1 / 148.9 |
| 3 | greedy/greedy | change2 | 2 | 24 | 0.149 | 0.160 | 0.161 | 0.032 | 0.656 | 95.34, 94.95 | 103.47, 103.03 | 0.161 | 28.7 / 159.6 |

`prefix check: 56 long texts, 420 newcomer texts, 0 mismatches -> PASS`

| c | arm | phases | aggregate tok/s per phase | mean | TTFT p50 s | TTFT p95 s | TTFT max s | inter-token p99 ms | admissions by K | stall marks | demotions |
|---:|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|
| 2 | base | 4 | 36.55, 36.09, 36.61, 36.14 | 36.35 | 0.301 | 0.483 | 0.490 | 308.5 | K3/cost-model x24 | 0 | 0 |
| 2 | change | 4 | 40.3, 40.68, 40.3, 40.7 | 40.49 | 0.401 | 0.486 | 0.487 | 231.6 | K3/cost-model x24 | 0 | 0 |
| 3 | base | 4 | 45.45, 44.77, 45.56, 44.81 | 45.15 | 0.825 | 2.445 | 2.455 | 438.0 | K3/cost-model x36 | 0 | 0 |
| 3 | change | 4 | 49.0, 49.87, 49.01, 48.58 | 49.11 | 0.452 | 0.835 | 0.837 | 441.4 | K3/cost-model x36 | 0 | 0 |
| 4 | base | 4 | 52.74, 53.38, 53.92, 53.71 | 53.44 | 1.924 | 2.706 | 3.022 | 463.0 | K3/cost-model x48 | 0 | 0 |
| 4 | change | 4 | 58.02, 58.15, 58.1, 58.22 | 58.12 | 0.604 | 0.913 | 1.066 | 459.3 | K3/cost-model x48 | 0 | 0 |

`text check: 216 requests, 54 (phase, prompt) groups, 0 with more than one text, 0 request errors -> PASS`

- Behind one stream the change acts as on GB10, in both forms: second token 0.472 to 0.002 s,
  newcomer end to end 0.94 to 0.44 s.
- Behind two streams the band admits the third request to plain decode (`K=0
  source=concurrency`, wave 3) while the two streams keep speculating, so its prime is in phase
  (b) on every arm: first token 0.82 to 0.86 s. That is memra#1244's ordering; neither form's
  newcomers' pass names a plain session. Behind three streams every session is plain.
- All tok/s per boot overlap across arms in every row. The spread between boots of one arm
  reaches 4 to 8% in the two-stream rows (base 83.6 to 87.3, final form 80.2 to 86.9 greedy;
  base 72.8 to 75.9, final form 72.7 and 72.8 sampled): the laptop card drifts between boots,
  and the pair run back to back (`base-6`, `change2-7`) reads 85.3 against 80.2 greedy and
  72.8 against 72.8 sampled. With two or three boots per arm that is no resolved difference in
  either direction; the request path of those rows is the base's.
- Every text is equal across the seven boots.

## CPU checks (dev desk, `devdesk/`)

On the change's final crates (`checks.log`): `cargo fmt --all -- --check`, `cargo clippy
--release --all-targets -- -D warnings`, the `memra-engine` lib suite (741 passed, 65 ignored:
device tests), the `memra-server` suite (1,206 passed, 14 ignored), the flags census and the
numeric-scope census all exit 0.

Red/green (`redgreen.log`, `tools/dd-redgreen.sh`, `tools/redarm.py`), three tests:
`spec_cold_serial_first_names_the_newcomers`, `spec_pass_steps_each_step_once`,
`spec_newcomers_prime_before_cohorts_is_wired`.

| arm | what it is | fails |
|---|---|---|
| `main` | the base's `worker.rs` with only the two pure functions and the tests added (phase (a) as before) | the wiring test |
| `oldorder` | `spec_cold_serial_first` names nobody | the naming test |
| `nojoin` | a session that primed first is never handled in the cohort pass | the pass test |
| `swapped` | the cohort pass runs first | the wiring test |
| `nopeer` | every newcomer is named, trimmed prime or not (the review's lone newcomer beside plain riders) | the wiring test |
| `noquantum` | the cohort pass does not read the prime quantum before the cohorts form | the wiring test |
| green | the change | none (3 of 3 pass) |

## Files

- `gb10/repro/`: the reproduction boot (rows, tick-trace server log, texts).
- `gb10/newcomer/`, `gb10/closed-loop/`, `gb10/stall-2long/`, `rtx5090/newcomer/`,
  `rtx5090/closed-loop/`: rows (`newcomer.jsonl`, `normal.jsonl`, `stall.jsonl`), `guard.log`
  (binary sha256, card, load at each boot), `boots.log`, the server logs gzipped (a row's
  `log_from`/`log_to` are byte offsets into the gunzipped log; the gzip level is the highest
  whose bytes do not trip the public-boundary scanner by chance), `texts/`.
- `gb10/newcomer-long/`: the long-prompt cell and the short-prompt cells on the final form.
- `gb10/inv/`, `gb10/bnd/`, `rtx5090/inv/` (final form), `gb10/inv-first-form/`,
  `gb10/bnd-first-form/`, `rtx5090/inv-first-form/`: gate rows, logs and texts.
- `devdesk/`: the CPU check and red/green logs.
- `tools/`: `newcomer_cell.py`, `stall_cell.py` (memra#1241's with `LONGS`), `summ_newcomer.py`,
  `summ_norm.py`, `summ_stall.py`, `dd-redgreen.sh` and `redarm.py` (the red arms).
