# A stop string ends a speculative request on the token that completes it (memra#1207)

Verdict: fixed. On the speculative routes a request that ended on a stop string counted its
completion tokens through the end of its decode burst, and its response stayed open while the
burst ran out. With the fix every stop row of `tools/spec-stop-string-gate.py` counts what plain
decode counts (13 of 13 per sampler and boot, greedy and temperature 1, at the default burst and
at `MEMRA_SPEC_BURST=2`, on one RTX 5090 Laptop GPU and on GB10), and closes at the level of the
non-stop rows. On the release head the same rows count to the burst's end (1 to 5 of 13 equal)
and close up to 665 ms (RTX 5090) and 1,309 ms (GB10) after their last visible chunk, against
34 and 75 ms for non-stop rows. Qwen3.8-27B NVFP4 + Q5_K, MTP and DSpark routes.

## The defect and the fix

`spec_burst_post` and its DSpark, GLM5 and gemma twins checked stop strings after the burst
returned, when every token of the burst was already in `generated` (what usage reports). The
HTTP layer cut the text at the stop string, so text and finish reason were right.

The check moved into `emit_spec_token_events`, the helper every speculative route emits
through. After each token it sends it runs the check plain decode runs (`contains_stop_string`
on the decoded bytes): the completing token is the last one sent and counted, and the terminal
latch closes. The round hook's continue-verdict then ends the burst at that round on MTP (serial
and pooled), DSpark (serial and pooled; the engine's commit hook now returns the verdict) and
GLM5. gemma emits at burst cadence with no round hook: its count is clamped and its burst still
runs to its target. The DeepSeek-V4 route checks stop strings per token before its commit and
was not affected. A `[spec-stop]` line names where each stop string completed.

The independent review found that a DSpark session that stopped inside a round (cache rows past
its fed tokens) was then refused at park as an invariant failure. A recurrent trunk parks
rewound to a prime boundary inside its fed tokens, so that shape now enters the park
(`dspark_park_shape_ok`). `review.md` has every finding and its disposition.

## Setup

- Qwen3.8-27B NVFP4 + Q5_K with its MTP head (`Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`). DSpark rows use
  the DFlash2 drafter at q4 with the 32,768-row rank trim (the RTX 5090 recipe).
- One RTX 5090 Laptop GPU; one DGX Spark (GB10). `MEMRA_CTX=8192`, `MEMRA_MAX_SESSIONS=2`, one
  request at a time, every request streamed with usage.
- Two GB10 units: the first ran the base and the 663379cd6a fix, the second the e3f4c1a478 fix
  (its own plain boot, `gb10/fix-e3f4c1a478/`), with the binary built there and copied to the
  first unit for its small-burst boot (same sha256). `session-gate` is byte-identical across the
  two fix heads on each device class (sha256 595c2329... on GB10, 232e5a13... on the RTX 5090), so
  its GB10 run covers both heads.
- Binaries. Base (the red arm): the release head 24b5f3e3f2, whose crates and tools equal main
  3de59bbf80's. Fix: built before the lane was rebased onto main, from 387141ae14 and 081a5d7101;
  their `crates/` trees equal 663379cd6a's and e3f4c1a478's on this branch (`git diff --quiet`),
  and the receipts are named by the rebased commits. Between 663379cd6a and e3f4c1a478 only the
  DSpark park predicate, the receipt lines on the GLM5, gemma and burst-cadence arms, comments
  and tests changed: the MTP path is the same code.
- Boots: `ref` (`MEMRA_SPEC_K=0`, plain decode), `spec` (the defaults: burst 32), `small`
  (`MEMRA_SPEC_BURST=2`), `dspark` (prefix cache off), `dspark-px` (prefix cache on, 4,000 MB,
  `MEMRA_MTP_SKIP=1`). One boot per GPU lease. The base and later fix runs on a device reuse that
  device's first plain boot (copied; the diff does not touch plain decode).
- Every row is one request, one run. Close times are single runs, not medians over repeats.

## Results

Stop rows: five one-word stop strings, three three-word ones, three requests with two stop
strings, one stop on the last token of `max_tokens`, one first turn of a conversation. "Equal to
the reference": text, finish reason and completion tokens equal plain decode's; on DSpark's
sampled rows (its rejection walk draws another stream than plain decode) the reference is the
route's own stream, where the stop string must complete on exactly the row's last token.

| run | boot | sampler | stop rows equal to the reference | close ms, stop rows median / max | close ms, non-stop median / max |
|---|---|---|---|---|---|
| RTX 5090 base 24b5f3e3f2 | spec (mtp) | t1 | 1/13 | 292 / 665 | 34 / 37 |
| RTX 5090 base 24b5f3e3f2 | spec (mtp) | greedy | 2/13 | 257 / 538 | 0 / 53 |
| RTX 5090 base 24b5f3e3f2 | small (mtp) | t1 | 4/13 | 112 / 241 | 34 / 123 |
| RTX 5090 base 24b5f3e3f2 | small (mtp) | greedy | 5/13 | 0 / 105 | 0 / 54 |
| RTX 5090 base 24b5f3e3f2 | dspark | t1 | 2/13 | 244 / 477 | 0 / 0 |
| RTX 5090 base 24b5f3e3f2 | dspark | greedy | 1/13 | 285 / 477 | 0 / 0 |
| RTX 5090 fix e3f4c1a478 | spec (mtp) | t1 | 13/13 | 36 / 90 | 34 / 36 |
| RTX 5090 fix e3f4c1a478 | spec (mtp) | greedy | 13/13 | 0 / 103 | 0 / 53 |
| RTX 5090 fix e3f4c1a478 | small (mtp) | t1 | 13/13 | 37 / 160 | 35 / 122 |
| RTX 5090 fix e3f4c1a478 | small (mtp) | greedy | 13/13 | 0 / 104 | 0 / 54 |
| RTX 5090 fix e3f4c1a478 | dspark | t1 | 13/13 | 0 / 98 | 0 / 0 |
| RTX 5090 fix e3f4c1a478 | dspark | greedy | 13/13 | 0 / 102 | 0 / 0 |
| RTX 5090 fix e3f4c1a478 | dspark-px | t1 | 12/12 | 0 / 98 | 0 / 32 |
| RTX 5090 fix e3f4c1a478 | dspark-px | greedy | 13/13 | 0 / 102 | 0 / 32 |
| GB10 base 24b5f3e3f2 | spec (mtp) | t1 | 1/13 | 485 / 1309 | 75 / 76 |
| GB10 base 24b5f3e3f2 | spec (mtp) | greedy | 2/13 | 509 / 1197 | 0 / 0 |
| GB10 base 24b5f3e3f2 | small (mtp) | t1 | 5/13 | 153 / 509 | 76 / 77 |
| GB10 base 24b5f3e3f2 | small (mtp) | greedy | 5/13 | 0 / 209 | 0 / 0 |
| GB10 fix 663379cd6a | spec (mtp) | t1 | 13/13 | 78 / 181 | 75 / 76 |
| GB10 fix 663379cd6a | spec (mtp) | greedy | 13/13 | 0 / 207 | 0 / 1 |
| GB10 fix 663379cd6a | small (mtp) | t1 | 13/13 | 77 / 334 | 75 / 78 |
| GB10 fix 663379cd6a | small (mtp) | greedy | 13/13 | 0 / 210 | 0 / 1 |
| GB10 (second unit) fix e3f4c1a478 | spec (mtp) | t1 | 13/13 | 77 / 181 | 75 / 78 |
| GB10 (second unit) fix e3f4c1a478 | spec (mtp) | greedy | 13/13 | 0 / 208 | 0 / 0 |
| GB10 (second unit) fix e3f4c1a478 | dspark-px | t1 | 13/13 | 0 / 88 | 0 / 0 |
| GB10 (second unit) fix e3f4c1a478 | dspark-px | greedy | 13/13 | 0 / 180 | 0 / 0 |
| GB10 fix e3f4c1a478 | small (mtp) | t1 | 13/13 | 77 / 334 | 76 / 76 |
| GB10 fix e3f4c1a478 | small (mtp) | greedy | 13/13 | 0 / 208 | 0 / 1 |

The base's counts are its burst ends: on the RTX 5090 at the default burst, temperature 1, 32,
66, 66, 96, 96 for the one-word rows where plain decode counts 19, 36, 50, 68, 72. A stop row
that closes later than a non-stop row on the fix is a stop string of several tokens: the server
holds back text that may start a stop string, so the last visible chunk arrives before the stop
string begins, in plain decode too. The judge allows max(3x, +100 ms) over the non-stop median
plus twice plain decode's own hold-back for the row. In the `dspark-px` sampled row the first
turn of the conversation is not judged on the route's own stream: its reference requests
restored the prompt from the prefix cache and ran on plain decode (DSpark admits cold sessions
only).

Per-row counts, texts and logs: each run's `runs.jsonl` and `judge.log`.

## Edge cases

Each is a coverage requirement of the gate, met per sampler on every fix device:

- A stop string that began in an earlier commit slice than it completed in (`spans_slices=1` in
  the `[spec-stop]` receipt): on the RTX 5090 fix, 5 (t1) and 3 (greedy) rows per MTP boot, 2 and
  2 per DSpark boot; on GB10, 5 and 3 (`spec`), 5 and 3 (`small`).
- Two stop strings in one burst (the pair's alone-counts in plain decode fall in one burst of the
  same request in that boot): 5 (t1) and 4 (greedy) requests over the MTP boots, on both devices.
  The first to complete ends the request, as in plain decode.
- A stop string only a drafted run the verify rejected would complete: candidates taken from the
  per-round drafts (`MEMRA_DEBUG_SPEC=1`) of a warm run of the free request, each confirmed in its
  own row's log (the same context, accepted count and rejected draft token). 4 (t1) and 8
  (greedy) confirmed rows per device; every one runs to `max_tokens` with plain decode's text.
- A stop on the last token of the budget: `budget` rows, `max_tokens` equal to plain decode's
  count for that stop string; finish `stop` and the same count on the fix.
- The next turn after a stop-ended turn: `turn2` rows equal plain decode's on every MTP boot and
  restore 1,248 prompt tokens from the prefix cache. On this model the MTP session pool frees a
  turn whose checkpoint is off the 32-token prime grid (`spec-park SKIPPED (grid law)`, stop or
  no stop), so the restore is the prefix cache's. The engine invariant behind it is
  `session-gate`'s new check: a continuation burst the hook stops after its second slice ends
  there with `committed` equal to the cache rows and to the true history less the pending token,
  and the next burst emits the oracle's tokens. MATCH at K=2 and K=3 on the RTX 5090 (both fix
  heads; the engine code is the same) and on GB10 (`session-gate-*.log`).
- DSpark park after a stop (the review's finding): with the prefix cache on, the 663379cd6a binary
  logs `dspark-park REFUSED ... budget-clamped overshoot` for 16 stop-ended sessions
  (`rtx5090/px-663379cd6a/`); e3f4c1a478 logs none. In the cold conversation (prefix cache on, `px3-*`: a first turn that ends on a stop string inside a round, then a second turn), the 663379cd6a binary refuses the stop-ended session at park (`cache pos 1265 != 1264 fed tokens`) and the second turn finds nothing in the pool; on e3f4c1a478 the session parks rewound to its prime boundary (1,216) and the second turn resumes it from the pool. A first pair of these runs (`px2-*`) stopped on a paragraph break that the answer never wrote in 96 tokens; the gate failed it as untested and the stop string changed. On GB10 the DSpark restore switch is on by default, so
  the pool parks nothing by design and the second turn restores 1,216 tokens from the prefix cache
  (`gb10/fix-e3f4c1a478/`, no park refusal).

## CPU tests

On the dev desk (nvcc 13.1, `MEMRA_CUDA_ARCH=120a`):

- Green on e3f4c1a478 (`cpu/green-e3f4c1a478/`): `cargo fmt --all -- --check` 0, `cargo test -p
  memra-server --no-fail-fast` 0 (lib: 1,187 passed, 11 ignored), `cargo clippy -p memra-server
  -p memra-engine --all-targets -- -D warnings` 0.
- Red (`cpu/red/`): the fix with the hook's stop-string latch removed (`red.patch`; main's hooks
  knew no stop strings) fails `stop_strings_end_every_spec_round_partition_on_the_completing_token`,
  `spec_burst_ends_at_the_round_that_completes_a_stop_string` and
  `dspark_burst_hook_ends_at_the_round_that_completes_a_stop_string`; they pass on the fix.
  `dspark_park_takes_a_recurrent_session_that_stopped_inside_a_round` asserts the park shape the
  old predicate (`pos == fed`) refused.

## What the gate itself learned

- The first GB10 `small` boot (`gb10/fix-663379cd6a-first/`) demoted a sampled session to plain
  decode (`cost model prefers plain`), and every later sampled row of that boot ran on plain
  decode: equal to the reference by construction. Speculative boots now run with
  `MEMRA_SPEC_GATE=0`, and a row that did not speculate fails. The rerun is `gb10/fix-663379cd6a/`.
- At `MEMRA_SPEC_BURST=2` the sampled rows yield no rejected-draft candidate on any device (most
  bursts there run one round, and the candidate picker skips a burst's first round, whose context
  tail it cannot rebuild). The default-burst boots confirm 4 per device. A DIR that holds only a
  small-burst boot (`gb10/fix-e3f4c1a478-small/`) therefore fails that one coverage line.
- DSpark's sampled rows cannot be compared with plain decode's sampled stream; the first DSpark
  boot (`rtx5090/fix-663379cd6a/`, judged by the gate before its own-stream rule) fails there for
  that reason only. The final DSpark boots are judged on the route's own stream.

## Not run

- GLM5 and gemma: no checkpoint of either is on the devices this lane used. Their count clamp is
  the shared helper's, covered by the unit tests; GLM5's verdict is the same loop change as
  DSpark's, not run on a device.
- Pooled cohorts: the gate sends one request at a time. A stopped member of a pooled cohort still
  gets its `Done` when the cohort call returns, as an EOS-ended member does on main (memra#1209).
- DSpark with the prefix cache off on GB10: DSpark on GB10 ran with the prefix cache on (the
  restore switch is on there by default, so the next turn restores from the prefix cache, not the
  pool; the gate accepts that path).

## Receipts

- `rtx5090/`: `base-24b5f3e3f2/` (red), `fix-663379cd6a/` and `fix-e3f4c1a478/` (the two fix
  heads), `px-663379cd6a/` (DSpark, prefix cache on, the first fix head), `px2-*`/`px3-*` (the
  cold conversation on both fix heads). `gb10/`: `base-24b5f3e3f2/`, `fix-663379cd6a/` (with
  `fix-663379cd6a-first/`, the demoted run), `fix-e3f4c1a478/` (second unit: plain, default
  burst, DSpark) and `fix-e3f4c1a478-small/`.
- `rtx5090/`, `gb10/`: one directory per gate DIR (`runs.jsonl`, `server-<boot>.log`,
  `env-<boot>.json`, `plan.json`, `reject-<boot>.json`, `texts/`, `judge.log`), and the
  `session-gate-*.log` runs. Machine paths are replaced by `$MODELS`, `$RUN` and `$HOME`.
- `cpu/`: the green and red logs. `review.md`: the review and its dispositions.
- `tools/summary.py`: the results table from the gate DIRs.
