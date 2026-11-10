# A sampled speculative burst carries its pending token (memra#1186, 2026-11-10)

Verdict: exact on every gate run so far, and not ready to land alone. With the carry every
seeded text is the base's: 54 of 54 rows of the seeded-invariance gate and 229 of 229 rows of
the new burst-boundary gate carry the base binary's text sha, on GB10 and on one RTX 5090 Laptop
GPU, and every one equals plain decode's. No sampled burst draws at its tail any more (0 against
1,186 on GB10) and 1,254 sampled bursts entered on a carried token. Two things stand between
this and a landing. The 12-boot timed cell on GB10 is owed (the device was lent out; the split
that named the two passes measured 10 to 12 ms of wall per sampled round). And the stall cell
shows that a session holding a carried token emits nothing while requests keep arriving with
the demotion gate off: greedy does that on main today, and the carry makes a sampled request do
it too, where it used to advance one token per burst. A bound on consecutive entry yields
(memra#1201) lands first.

Model: Qwen3.8-27B NVFP4 + Q5_K GGUF with its MTP head, context 8,192, two sessions, thinking
off, every switch at the device default unless a cell says otherwise. Devices: GB10, one DGX
Spark (sm_121a), and one RTX 5090 Laptop GPU (sm_120a). Arms are two binaries per device:
`base`, main 2628ece940 plus the no-round fix (memra#1196, commit e76ef8429d, the candidate's
parent), and `cand`, this change. `main` in the stall cell is 2628ece940 itself. Both arms of a
device are built on the same machine with the same toolchain; every run directory's `guard.log`
or the gate's first lines hold the binary's sha256.

The rule and the decision: [`docs/decisions/SAMPLED-PENDING-CARRY.md`](../../docs/decisions/SAMPLED-PENDING-CARRY.md).
Where the two passes were found: [`research/gb10-sampled-round-split-20261110/`](../gb10-sampled-round-split-20261110/).

## What changed

A speculative burst ends on a pending token: emitted, not yet in the caches. The sampled tail
committed it with a solo T=1 trunk pass and drew the next token from that pass's logits; the
next burst fed that token with a second solo pass. The sampled tail now stashes the pending
token on the session, as the greedy tail has since 2026-08-01, and the next burst's round 0
verifies it as column 0 and draws the next position there, at the same position-keyed event
(memra#1154), from the same row, in the same penalty window. Two rollback seams keep the old
tail (`MEMRA_SPEC_SAMPLED_BOUNDARY=0`, and `MEMRA_SPEC_PEN_SESSION=0` on a penalised request).
The context-full check counts a sampled session's pending token, so a sampled request that
fills its context stops where it stopped before.

## The texts are the base's

`tools/seeded-invariance-gate.py`: a plain reference boot, then `MEMRA_SPEC_K` pinned 0, 1, 2, 3
(lone prompts, a pair, a resumed turn, the parked-prompt pair) and a forced demotion. Ten
prompts, 240-token answers, 54 rows. Each binary must pass on its own, and the two row files
are compared row by row (`tools/rowcmp.py`).

| device | options | base | candidate | rows with the base's text sha | same completion, prompt and cached tokens |
|---|---|---|---|---:|---:|
| GB10 | `--penalised device` (the device penalty arm is the default there on this main) | PASS, 54 rows | PASS, 54 rows | 54 of 54 | 54 of 54 |
| RTX 5090 | default | PASS, 54 rows | PASS, 54 rows | 54 of 54 | 54 of 54 |

Acceptance per drafted token moves a little, because the token after a boundary is now drafted
and verified in a round where it used to be drawn outside one. Mean over the gate's ten rows
per K, base then candidate: GB10 K=1 0.685 and 0.691, K=2 0.542 and 0.549, K=3 0.425 and 0.426;
RTX 5090 0.680 and 0.684, 0.531 and 0.534, 0.418 and 0.420.

The demotion rows: all four requests of each binary log their handoff at event = generated + 1
and write the reference text. With the carry the handed token is the flush's draw
(`spec_flush_pending` with the request's sampler); before, the tail's.

A first pass on the commit this lane started from (9ca50c3556, before tonight's landing moved
main): GB10, the same gate, 54 of 54 rows equal between that base and the candidate of that
day, the host-drawn penalised rows included (`gb10/earlier-base/`). It is superseded by the
rows above and kept as a second base.

Owed on the rebased arms: the RTX 5090 runs with the 27B's trimmed drafter (base: PASS, 40
rows; candidate not run yet) and with the 9B and its own-trim drafter.

## A request that ends on a burst boundary

`tools/spec-burst-boundary-gate.py` (new, `docs/TESTING.md`): eleven answers that end on EOS, a
JSON-object answer, `max_tokens` 31, 32, 33, 63, 64, 65, five stop strings, a resumed second
turn, the parked-prompt pair, a long answer under arriving requests with the demotion gate off
and at its default, and three open answers that run into a 640-token window; temperature 1,
temperature 0.7 with top_p 0.95, a frequency penalty, greedy. Four speculative boots (the
device's own K and burst; K=1 burst 2; K=3 burst 2; K=3 burst 3). 229 rows.

| | GB10 base | GB10 candidate | RTX 5090 base | RTX 5090 candidate |
|---|---|---|---|---|
| verdict | PASS (`--carry no`) | PASS | PASS (`--carry no`) | PASS |
| every speculative text equals plain decode's, same finish reason | yes | yes | yes | yes |
| rows with the base's text and finish reason | | 229 of 229 | | 229 of 229 |
| boundary draws at a sampled burst's tail | 1,186 | 0 | 1,160 | 0 |
| sampled bursts that entered on a carried token | 0 | 1,254 | 0 | 1,222 |
| yield boot: continuation bursts that ran no round, and how many on a carried token | 23, 0 | 24, 24 | 13, 0 | 13, 13 |
| load boot: demotions to plain decode | 2 | 2 | 2 | 2 |

The base passes only under `--carry no`, which reports the two carry lines instead of judging
them ("1186 sampled bursts committed their pending token with a tail pass", "no sampled burst
entered on a carried token"): those are the gate's teeth, and under the default they fail.

Where the last token of a request sat, candidate (rows whose last token is the last of a burst
/ the first of the next one): GB10 EOS sampled 4 / 3, greedy 6 / 3; stop string sampled 3 / 8,
greedy 4 / 9; budget sampled 9 / 2, greedy 9 / 9. RTX 5090 EOS 2 / 8 and 12 / 3; stop string
4 / 5 and 7 / 7; budget 13 / 7 and 9 / 10. Every end kind is met on a boundary, sampled and
greedy, on both devices.

The context edge (an open request in a 640-token window; plain decode writes 602 tokens): the
speculative text is a prefix of plain decode's, and base and candidate stop at the same token:
GB10 600, 599, 600 (temperature 1, penalised, greedy), RTX 5090 597, 599, 599. Without the
pending token counted in the context-full check a sampled request would run one more round
here.

Stop strings and `completion_tokens`. Of the 229 rows, completion tokens differ between base
and candidate on 3 (GB10) and 8 (RTX 5090), all of them sampled stop-string rows in the
small-burst boots, each with the same text and the same finish reason (`stop`) in both arms;
for example 38 against 36, 31 against 33, 49 against 53. Neither count is the stop's. On the
speculative route a stop string is checked after the burst, so the count runs to the end of the
burst in which the string completed, and the carry moves a small burst's end by a token or two.
At the default burst the two arms agree on every stop row and both are a burst end: plain
decode / base / candidate on GB10 at temperature 1 are 25 / 33 / 33, 31 / 33 / 33, 36 / 67 / 67,
42 / 67 / 67, 49 / 67 / 67, and greedy 20 / 34 / 34 through 49 / 66 / 66. That over-count is on
main and is not this change's; it is reported on its own.

## The stall cell

`tools/stall_cell.py`: one long streamed request runs alone for 4 s, then four closed-loop
clients send one-token requests back to back for 20 s, then it runs alone for 8 s. RTX 5090,
rates are tokens per second on the client's clock, no timing claim on this card.

| binary, demotion gate | class | tokens before / in the window / after | rate in the window | longest gap in the window | no-round bursts, of them silent |
|---|---|---|---:|---:|---|
| main, gate off | greedy | 204 / 3 / 408 | 0.14 | 20.8 s | 23, 23 |
| main, gate off | temperature 1 | 162 / 24 / 360 | 1.16 | 0.93 s | 22, 0 |
| base, gate off | greedy | 204 / 3 / 411 | 0.14 | 20.8 s | 23, 23 |
| base, gate off | temperature 1 | 162 / 24 / 360 | 1.15 | 0.93 s | 22, 0 |
| candidate, gate off | greedy | 207 / 2 / 410 | 0.10 | 20.7 s | 23, 23 |
| candidate, gate off | temperature 1 | 164 / 2 / 384 | 0.10 | 20.7 s | 23, 23 |
| main, default gate | greedy | 207 / 118 / 249 | 5.85 | 0.30 s | demoted to plain decode |
| main, default gate | temperature 1 | 164 / 118 / 250 | 5.84 | 0.33 s | demoted to plain decode |

A burst that begins while a request waits for admission runs no round. Without a carried token
it still emits its boundary token; on a carried token it emits nothing. So with the demotion
gate off a greedy request stalls for as long as requests keep arriving, on main today, and the
carry gives a sampled request the same stall in place of a one-token-per-burst crawl. With the
gate at its default the request is handed to plain decode and advances. The bound on
consecutive entry yields is its own change (memra#1201) and lands before this one.

## The no-round fix in serving (memra#1196)

Banked here for memra#1196: `rtx5090/noround/` (`tools/noround_cell.py` of
`research/spec-no-round-carried-token-20261110/`, its last form: the long request alone for
1.5 s, then two one-token clients for 3 s; greedy, `MEMRA_SPEC_GATE=0`).

| binary | request | max_tokens | completion tokens, finish | continuation bursts, no-round on a carried token | next request: prompt / cached |
|---|---|---:|---|---|---|
| main 2628ece940 | chat turn, alone | 400 | 400, length | 12, 0 | 1,718 / 1,248 |
| main 2628ece940 | chat turn, arrivals | 400 | 398, length | 19, 7 | 1,715 / 1,248 |
| fix | chat turn, alone | 400 | 400, length | 12, 0 | 1,718 / 1,248 |
| fix | chat turn, arrivals | 400 | 400, length | 21, 8 | 1,717 / 1,248 |
| main and fix | text completion, arrivals | 400 | 286, stop (EOS) | 16, 7 | 1,407 / 1,088 |

On main the request with arrivals ends two tokens short of its budget: each no-round burst on
a carried token put one more token into the committed list, and the context-full check reads
that list against a session sized to the request. With the fix it writes its 400. The next
turn's restore is the same in every row. On a 640-token window the open request writes 605
tokens under arrivals and 606 alone on both binaries: the speculative context-full stop is
taken at a burst's end, so its token count moves with the burst ends; that is not the defect.

## Owed and not determined

- The timed cell on GB10 (12 boots, base cand cand base three times, the classes of
  `research/spec-draft-accept-leftovers-20261110/`): owed. The unit was lent to another project
  for the evening.
- The timed cell on the RTX 5090: owed. The laptop was off its performance power mode during
  this lane, so only pass/fail gates ran there.
- On the rebased arms, not run yet: the end-of-request cell (`tools/close_cell.py`: text, usage
  and the wall from the last chunk to the end of the response, after EOS, a budget, a stop
  string, a resumed turn, a park, a forced demotion and a hang-up), the two rollback seams
  (`tools/seam_rows.py`), the stall cell on GB10, `session-gate`, the hit gate with its teeth
  arm, `sample_check`, `kernel-check`, the trimmed-drafter rows on the candidate.
- Not measured: the RTX PRO 6000, three or more sampled requests at once, a multi-head MTP
  chain, the DFlash2, GLM5, DSpark and gemma assistant-drafter routes (untouched).

## Method

Dev desk (no GPU), the candidate's crates: `cargo fmt --all -- --check`, `cargo clippy --release
--all-targets -- -D warnings`, `cargo test --release -p memra-engine --lib`, `cargo test
--release -p memra-server` (1,162 passed in the lib suite), `tools/check-flags.sh`,
`tools/check-numeric-scope.py`: green.

GB10: both servers built on the device (`tools/gb10-build.sh`, `cargo build --release --locked
-j 12 -p memra-server --bin memra-server`); one gate run or one boot per device lease
(`tools/cells_driver.sh` is the driver's shape; `tools/gb10-gate.sh`, `tools/gb10-boot.sh` run
on the device). The boundary gate there ran with `--env MEMRA_TIMEOUT_MS_MAX=600000` on both
arms: its requests are not streamed and the device is slow enough for the 90 s non-streaming
deadline to matter under arrivals. RTX 5090: binaries built on a CPU box for sm_120a, one gate
run per card lease (`tools/gates_5090.sh`).

## Files

- `gb10/inv-base/`, `gb10/inv-cand/`, `rtx5090/inv-base/`, `rtx5090/inv-cand/`: the invariance
  gate (`runs.jsonl`, the gate's log, the server logs). `gb10/earlier-base/`: the two row files
  of the first pass.
- `gb10/boundary-base/`, `gb10/boundary-cand/`, `rtx5090/boundary-base/`,
  `rtx5090/boundary-cand/`: the burst-boundary gate (`runs.jsonl` with each request's burst
  ends, the stop strings, the server logs).
- `rtx5090/trim27-base/`: the trimmed-drafter rows of the base.
- `rtx5090/stall/`: `stall.jsonl` and the four server logs.
- `rtx5090/noround/`: the no-round serving cell, main and fix (`resume.jsonl`, `ctx.jsonl`, the
  server logs).
- `tools/`: `rowcmp.py`, `stall_cell.py`, `close_cell.py`, `close_summary.py`, `seam_rows.py`,
  `ab_cell.py` and `carry_summary.py` (the timed cell and its summary), `thermal.py`,
  `export.py` (wrote these directories), `gb10-build.sh`, `gb10-gate.sh`, `gb10-boot.sh`,
  `cells_driver.sh`, `gates_5090.sh`.
