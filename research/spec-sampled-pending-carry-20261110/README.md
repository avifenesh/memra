# A sampled speculative burst carries its pending token (memra#1186, 2026-11-10)

Verdict: exact on every gate run on the RTX 5090, and on GB10 so far; GB10 pending: the rows marked below and the timed cell.

Model: Qwen3.8-27B NVFP4 + Q5_K GGUF with its MTP head, context 8,192, thinking off, every
switch at the device default unless a cell says otherwise. Devices: GB10, one DGX Spark
(sm_121a), and one RTX 5090 Laptop GPU (sm_120a). Every cell is pass/fail style with single
runs, except the timed cell on GB10.

Arms, two binaries per device, both on main 66ee31fcb6 (v0.141.1 plus memra#1216, #1202,
#1215, #1219, so the entry-yield bound memra#1201 and the stop-string fix memra#1207 are in
both): `main`, that commit, and `cand`, that commit plus this change. RTX 5090: both built on
one CPU box with one toolchain (sha256 `702eae9c...` and `7c6661fd...`). GB10: `main` is the
sm_121a build the landing of 83d37f5179 qualified (66ee31fcb6 adds receipts only; sha256
`0f20018a...`), `cand` is built on the device (`31d1a420...`). Every run directory's
`guard.log` or the gate's first lines hold the binary's hash. The change was rebased once more
before review, onto ede1b8119f (memra#1208, the prime boundary's publication into the prefix
cache, which does not touch the burst path); the final head's own rows are in
[the final head](#the-final-head).

The rule and the decision: [`docs/decisions/SAMPLED-PENDING-CARRY.md`](../../docs/decisions/SAMPLED-PENDING-CARRY.md).
Where the two passes were found: [`research/gb10-sampled-round-split-20261110/`](../gb10-sampled-round-split-20261110/).
The bound this change runs under: [`research/spec-entry-yield-bound-20261110/`](../spec-entry-yield-bound-20261110/).

## What changed

A speculative burst ends on a pending token: emitted, not yet in the caches. The sampled tail
committed it with a solo T=1 trunk pass and drew the next token from that pass's logits; the
next burst fed that token with a second solo pass. The sampled tail now stashes the pending
token on the session, as the greedy tail has since 2026-08-01, and the next burst's round 0
verifies it as column 0 and draws the next position there, at the same position-keyed event
(memra#1154), from the same row, in the same penalty window. Two rollback seams keep the old
tail (`MEMRA_SPEC_SAMPLED_BOUNDARY=0`, and `MEMRA_SPEC_PEN_SESSION=0` on a penalised request).
The context-full check counts a sampled session's pending token, so a sampled request that
fills its context stops where it stopped before. The worker's two flushes of a pending token
(a demotion to plain decode, a park at the end of a request) draw the token after it with the
request's own sampler (`spec_sampled_pending_carry_is_wired` checks both and the context-full
count; it fails on main, where the park flush passes no sampler).

## The entry-yield bound and the carry

The bound (`SPEC_ENTRY_YIELD_MAX = 1` in the worker) acts on a burst whose first commit hook
carries no token while a request waits for admission: such a session yields silently at most
once in a row, and at its next such entry runs one round. It reads whether the hook call
carries a token, not the sampling class. A burst carries no token at its entry exactly when it
enters on a carried pending token: the engine pushes the boundary token into the burst's
output before that hook only when nothing is carried. So with the carry a sampled burst that
enters on a carried token takes the bound as a greedy one does, and on the two rollback seams,
which keep the tail pass, a sampled burst still enters on its drawn boundary token, which the
entry hook emits, and the bound does not act. Before the bound, the carry turned a sampled
request's one-token-per-burst crawl under arrivals into a full stall
(`prerebase/rtx5090/stall/`); the stall cell below shows it advancing under the bound.

## The texts are main's

`tools/seeded-invariance-gate.py`: a plain reference boot, then `MEMRA_SPEC_K` pinned 0, 1, 2, 3
(lone prompts, a pair, a resumed turn, the parked-prompt pair) and a forced demotion. Ten
prompts, 240-token answers, 54 rows, the device penalty arm at its default on both devices.
Each binary must pass on its own, and the two row files are compared row by row
(`tools/rowcmp.py`).

| device | main | candidate | rows with main's text sha | same completion, prompt and cached tokens |
|---|---|---|---:|---:|
| GB10 | PASS, 54 rows | PASS, 54 rows | 54 of 54 | 54 of 54 |
| RTX 5090 | PASS, 54 rows | PASS, 54 rows | 54 of 54 | 54 of 54 |

Acceptance per drafted token moves a little, because the token after a boundary is now drafted
and verified in a round where it used to be drawn outside one. Mean over the gate's ten rows
per K, main then candidate: GB10 K=1 0.685 and 0.691, K=2 0.542 and 0.549, K=3 0.425 and
0.426; RTX 5090 0.680 and 0.684, 0.531 and 0.534, 0.418 and 0.420.

## A request that ends on a burst boundary

`tools/spec-burst-boundary-gate.py` (new, `docs/TESTING.md`): eleven answers that end on EOS, a
JSON-object answer, `max_tokens` 31, 32, 33, 63, 64, 65, five stop strings, a resumed second
turn, the parked-prompt pair, a long answer under arriving requests with the demotion gate off
and at its default, and three open answers that run into a 640-token window; temperature 1,
temperature 0.7 with top_p 0.95, a frequency penalty, greedy. Four speculative boots (the
device's own K and burst; K=1 burst 2; K=3 burst 2; K=3 burst 3). 229 rows. Every speculative
row must equal plain decode's text, finish reason and completion tokens, stop-string rows
included since memra#1207.

| | GB10 main | GB10 candidate | RTX 5090 main | RTX 5090 candidate |
|---|---|---|---|---|
| verdict | PASS (`--carry no`) | GB10 pending | PASS (`--carry no`) | PASS |
| rows with main's text, finish reason and completion tokens | | GB10 pending | | 229 of 229 |
| boundary draws at a sampled burst's tail | 1,179 | GB10 pending | 1,157 | 0 |
| sampled bursts that entered on a carried token | 0 | GB10 pending | 0 | 1,222 |
| yield boot: continuation bursts that ran no round, and how many on a carried token | 23, 0 | GB10 pending | 13, 0 | 13, 13 |
| load boot: demotions to plain decode | 2 | GB10 pending | 2 | 2 |

`--carry no` reports the two carry lines on main instead of judging them; under the default
they fail there, which is the gate's teeth. Before the rebase, on a main without memra#1207,
the counts of 3 (GB10) and 8 (RTX 5090) sampled stop-string rows in small-burst boots differed
between the arms, each counting to its burst's end; with the stop-string fix every count is
plain decode's in both arms.

Where the last token of a request sat, candidate (rows whose last token is the last of a burst
/ the first of the next one), RTX 5090: EOS sampled 2 / 8, greedy 12 / 3; stop string 6 / 5
and 11 / 7; budget 13 / 7 and 9 / 10. GB10: pending. Every end kind is met on a boundary,
sampled and greedy.

## Stop strings at temperature 1

`tools/spec-stop-string-gate.py` (memra#1207's gate) on the candidate: the plain reference
boot, the default burst and `MEMRA_SPEC_BURST=2`, greedy and temperature 1 rows, MTP route.

| device | boot | t1 stop rows equal to plain decode | greedy | coverage |
|---|---|---:|---:|---|
| RTX 5090 | default | 13 of 13 | 13 of 13 | met |
| RTX 5090 | burst 2 | 13 of 13 | 13 of 13 | met |
| GB10 | default, burst 2 | GB10 pending | | |

Coverage per sampler on the RTX 5090: stop strings spanning commit slices 10 (t1) and 6
(greedy), two stop strings in one burst 4 and 4, confirmed rejected-draft stops 8 and 8. The
temperature 1 non-stop rows close a median of 0 ms after their last chunk on both boots: no
closing tail pass.

## The bound holds for the sampled class

`research/spec-entry-yield-bound-20261110/tools/stall_cell.py`, the bound's own cell: one long
streamed request at temperature 1 (seed 611) runs alone for 8 s, so it holds a carried token;
then N closed-loop clients send one-token requests for 30 s with a think time; then it runs
alone for 12 s. One boot per arm, four cells per boot.

RTX 5090, `MEMRA_SPEC_GATE=0` (the bound's setting on this card: with the gate at its default
the band admits the short requests to plain decode and no stall arises):

| clients | think ms | long tokens in 30 s, main -> candidate | longest gap s | silent no-round bursts | short p50 s |
|---:|---:|---|---|---|---|
| 2 | 0 | 64 -> 73 | 0.49 -> 0.95 | 0 -> 32 | 0.481 -> 0.448 |
| 2 | 200 | 115 -> 131 | 0.48 -> 0.72 | 0 -> 60 | 0.318 -> 0.295 |
| 4 | 0 | 35 -> 40 | 0.93 -> 1.84 | 0 -> 17 | 0.927 -> 0.906 |
| 4 | 200 | 44 -> 50 | 0.93 -> 1.61 | 0 -> 22 | 0.730 -> 0.738 |

GB10, default gate: GB10 pending.

- The candidate advances in every cell. Its worst gaps match the bound's greedy rows on the
  same card and setting (0.95 s at two clients and 0 ms, 0.72 s at 200 ms, 1.84 s at four
  clients), as they should: a sampled session that holds a carried token now takes the
  greedy path through the bound.
- Main's sampled request never enters on a carried token, so it emits its boundary token at
  every entry (0 silent bursts) and crawls with a smaller gap. The candidate yields silently
  once, then runs a round that emits several tokens, and writes 9 to 16 more tokens in the
  window.
- Every long text is prefix-equal across arms and cells (the stream is cut at a time).

## The end of a request

`tools/close_cell.py`: per class (greedy as the control, temperature 1, temperature 0.7 with
top_p 0.95, penalised), streamed, one request at a time: EOS, a budget, a stop string, a
two-turn conversation, a park and its exact-prompt follow-up, another prefix, a hang-up after
40 chunks and the same request again; and with `MEMRA_SPEC_DEMOTE_AT=40` the turn and park
pairs demoted. Judged: text sha, finish reason, prompt, cached and completion tokens equal
between the arms. Reported: `close_ms`, the wall from the last content chunk to the end of the
response.

| device | mode | rows | judged differences | sampled close ms, main -> candidate | greedy close ms |
|---|---|---:|---:|---|---|
| RTX 5090 | close | 40 | 0 | 33.9 to 36.7 -> 0.06 to 0.14 (one row 89.6 -> 53.5) | unchanged |
| RTX 5090 | demote | 16 | 0 | park1 33.8 to 34.7 -> 0.07 to 0.15; the rest end in plain decode, 32 in both | unchanged |
| GB10 | close, demote | GB10 pending | | | |

A sampled request used to end one solo pass after its last chunk; with the carry its last token
stays pending, and the session is flushed only when it is parked, after the response has
ended.

## The rollback seams and the hit gate (RTX 5090)

`tools/seam_rows.py`: one boot per seam setting (`MEMRA_SPEC_SAMPLED_BOUNDARY=0`,
`MEMRA_SPEC_PEN_SESSION=0`, both), five lone sampled requests of 240 tokens each, three of them
unpenalised and two penalised. Both seams define their text by the burst boundary, so what
must hold is main's text under each. 15 of 15 rows carry main's text sha and completion
tokens. Where a seam keeps the tail pass (12 rows: every request under the boundary seam and
under both, the two penalised ones under the penalty seam) the candidate also has main's
rounds and no burst carries. The three unpenalised requests under the penalty seam carry (7
carried entries each, against 8 tail draws on main), with main's text.

`tools/spec-on-cache-hit-gate.sh qwen` (Qwen3.5-9B and its own-trim drafter), whose sampled
boundary checks this change updates (a continuation burst has no boundary site now; sampled
bursts must enter on a carried token and none may draw at its tail): ALL GREEN. Its teeth arm
(`MEMRA_HITGATE_TEETH=1`, the boundary door shut, so no burst may carry): ALL GREEN.

## session-gate

`session-gate` with `MEMRA_SPEC_K=3` on the candidate: four turns (cold, resumed, rewound, a
continuation burst), a no-round burst on a carried token (memra#1196), a burst the hook stops
mid-way (memra#1207), and the sampled carry check: one seeded, frequency-penalised request in
one burst, burst after burst, and with a no-round burst after every burst; every burst must end
on a carried token, `committed` must stay the cache rows, and the three must emit the same
tokens. RTX 5090: ALL TURNS MATCH ("sampled carry: 72 tok in one burst, 76 burst after burst,
76 with no-round bursts, first 72 equal true"). GB10: pending.

## The timed cell on GB10

GB10 pending: 12 boots, main cand cand main three times, one boot per lease, after the pass/fail cells.

## The final head

Pending: the CPU suites, session-gate and the seeded-invariance gate on the frozen head's build.

## Before the rebase

`prerebase/`: the same gates on main 2628ece940 plus memra#1196's fix (`base`) against the
carry of that day, before the bound and the stop-string fix landed. Seeded-invariance 54 of 54
rows equal on both devices; burst-boundary 229 of 229 texts and finish reasons equal, with
1,186 (GB10) and 1,160 (RTX 5090) tail draws on the base and none on the carry; the stall cell
(RTX 5090, gate off, four clients for 20 s) where the carry stalled a sampled request (2 tokens
in the window, longest gap 20.7 s, against 24 on main); and the no-round serving rows banked for
memra#1196 (`prerebase/rtx5090/noround/`). `prerebase/gb10/earlier-base/` is a first pass on
9ca50c3556.

## Not covered

- The RTX 5090 timed cell: only pass/fail cells ran on that card. The mechanism is in shared
  code; the price of the two passes differs per device.
- The RTX PRO 6000, three or more sampled requests at once, a multi-head MTP chain.
- The routes that still use the rejection walk (DFlash2, GLM5, DSpark, the gemma assistant
  drafter): no pending carry for a sampled request there, unchanged.

## Method

Dev box (no GPU), the candidate's crates: `cargo fmt --all -- --check`, `cargo clippy --release
--all-targets -- -D warnings`, `cargo test --release -p memra-engine --lib` (739 passed), `cargo
test --release -p memra-server` (1,191 passed in the lib suite), `tools/check-flags.sh`,
`tools/check-numeric-scope.py`: green. `spec_sampled_pending_carry_is_wired` on main with the
test added: FAILED ("a worker flush draws with the request's own sampler: (&engine, &mut sess,
None)").

GB10: one gate run or one boot per device lease. The boundary gate there runs with `--env
MEMRA_TIMEOUT_MS_MAX=600000` on both arms: its requests are not streamed and the device is slow
enough for the 90 s non-streaming deadline to matter under arrivals. RTX 5090: one gate run per
card lease.

## Files

- `gb10/`, `rtx5090/`: `inv-main/`, `inv-cand/` (the invariance gate), `boundary-main/`,
  `boundary-cand/` (the burst-boundary gate), `stop/` (the stop-string gate's boots and
  `judge.log`), `stall/` (`stall.jsonl`, `summary.md`, `prefix.txt`, the server logs),
  `close/` (`close.jsonl`, `demote.jsonl`, the summaries, the server logs), `sg-cand/` or
  `sg-cand.log` (session-gate), and on GB10 `perf/` (the timed cell).
- `prerebase/`: the rows before the rebase, above.
- `tools/`: `rowcmp.py`, `stall_cell.py` (the pre-rebase stall cell), `close_cell.py`,
  `close_summary.py`, `seam_rows.py`, `ab_cell.py` and `carry_summary.py` (the timed cell and
  its summary), `thermal.py`, `export.py` (wrote these directories), and the device drivers of
  the pre-rebase runs.
