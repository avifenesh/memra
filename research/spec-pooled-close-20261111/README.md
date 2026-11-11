# A pooled cohort member closes at the round its stream ended (memra#1209)

Verdict: fixed. When several sessions speculate in one pooled cohort (MTP or DSpark), a member
whose stream ends on a stop string, EOS or its `max_tokens` now gets its usage and `[DONE]` at the
round its burst ended, while the cohort runs on for the others. On main it got them only when the
whole cohort call returned. A member that stopped on a stop string closed 293 / 863 ms (p50, cohort
of 2 / 4) after its last visible chunk on the RTX 5090 MTP route and 619 / 1,382 ms on GB10, up to
2,498 ms; with the final head 36 / 38 ms and 77 / 81 ms, the level of the same requests alone (35
ms on the RTX 5090; on GB10 37 ms over both samplers, 74 ms at temperature 1). Texts, finish
reasons and usage are unchanged: every member 0 equals plain decode (on DSpark at temperature 1,
its own solo stream), and every companion wrote one text in every cohort, on main and on the fix.
Aggregate throughput of a four-member cohort: fix/main 0.992 (mean of six interleaved pairs, range
0.978 to 1.017, both samplers) on the RTX 5090, and 1.009 (t1) and 1.002 (greedy) on GB10 (three
pairs each, range 0.999 to 1.023); the spread inside one boot is up to 4%, so no difference above
about 1% is detectable. Qwen3.8-27B NVFP4 + Q5_K with its MTP head, and the DFlash2 drafter on the
DSpark route; one RTX 5090 Laptop GPU, two DGX Spark (GB10) units.

## The defect and the fix

`step_spec_pooled` and `step_dspark_pooled` ran each member's post step (`spec_burst_post`,
`dspark_burst_post`) and with it `finish`, which sends the token snapshot and `Event::Done`, after
the cohort call returned. Since memra#1207 a member that stops ends its own rounds at its round
hook's verdict, but its close waited for the slowest member's burst.

The pooled drivers now call a per-member end hook at the gate pass where the member's outcome
forms (the round after its last). The worker's hook decides through `spec_burst_verdict`, the post
step's verdict in one function, and sends what `finish` sends, from the values the post step writes
back later. The post step, park, publication and retire still run after the cohort returns;
`finish` then sends nothing and compares what it computed with what was sent. Why this design and
not the heavier one: `docs/decisions/POOLED-MEMBER-CLOSE.md`. A context-full end and the DSpark
burst-cadence arm (`MEMRA_SPEC_FIRST_TOKEN_EAGER=0`) keep the old close.

## Setup

- `tools/spec-stop-string-gate.py --cohort 2,4` (`docs/TESTING.md`). Member 0 is one of the gate's
  requests: five one-word and three three-word stop strings taken from plain decode's answer, an
  EOS answer and a 96-token answer. One or three companions (other prompts, no stop string, 192
  tokens) are sent with it. Every request is streamed with usage, greedy and at temperature 1
  (seeded). Each boot also runs member 0's requests alone (the solo rows) and three throughput
  cohorts of four companions per sampler. `MEMRA_CTX=8192`, `MEMRA_MAX_SESSIONS=4`.
- MTP boots run with `MEMRA_SPEC_GATE=0`. DSpark boots use the RTX 5090 recipe (q4 drafter,
  32,768-row rank trim, prefix cache off) with the gate on, `MEMRA_SPEC_GATE_LOW=4` and
  `MEMRA_SPEC_GATE_HIGH=8` (see "What the gate learned").
- Binaries, all release builds of this repo:
  - main ede1b8119f, built with the same toolchain as the fix on each device (RTX 5090: on the
    build server with nvcc 13.1; GB10: on the unit). GB10 main's close was measured with main's
    qualification-cell binary of 29e3632d87, whose crates equal ede1b8119f's.
  - fix 97f0364583 (`fix`, `fix2`, `fix-dspark` boots, the throughput A/B #1, seeded-invariance,
    burst-boundary, session-gate on the RTX 5090; `fix`, `fix-dspark`, throughput and
    seeded-invariance on GB10 from a tree with the same code plus a test-only change).
  - final head f47c2092a4 (`fix3` boots, throughput A/B #2). The review's changes after
    97f0364583 touch only the worker's host side; the session-gate binary is byte-identical.

## Close, main against the fix

Stopped member 0: last visible chunk to end of response. Solo is the same boot's solo rows that end
without a stop string. MTP rows: 16 per cell (5 one-word and 3 three-word stop strings, two
samplers); DSpark rows: 9 to 11 (at temperature 1 the route writes its own stream, which does not
always reach plain decode's stop words).

| device, route | binary | solo p50 | cohort of 2: p50 / max | cohort of 4: p50 / max |
|---|---|---|---|---|
| RTX 5090, MTP | main | 35 ms | 293 / 899 ms | 863 / 2,011 ms |
| RTX 5090, MTP | f47c2092a4 | 35 ms | 36 / 179 ms | 38 / 304 ms |
| RTX 5090, DSpark | main | 0 ms | 126 / 597 ms | 271 / 1,276 ms |
| RTX 5090, DSpark | f47c2092a4 | 0 ms | 0 / 169 ms | 2 / 103 ms |
| GB10, MTP | main | 38 ms | 619 / 1,715 ms | 1,382 / 2,498 ms |
| GB10, MTP | 97f0364583 code | 42 ms | 80 / 343 ms | 83 / 530 ms |
| GB10, DSpark | main | 0 ms | 90 / 853 ms | 88 / 1,683 ms |
| GB10, DSpark | 97f0364583 code | 0 ms | 1 / 180 ms | 4 / 180 ms |
| GB10 (unit-b), MTP | f47c2092a4 | 37 ms | 77 / 334 ms | 81 / 373 ms |
| GB10 (unit-b), DSpark | f47c2092a4 | 0 ms | 0 / 217 ms | 18 / 322 ms |

The maxima on the fix are stop strings that plain decode also holds back (text that may begin the
stop string is held from the stream until it completes): in a cohort those tokens arrive a cohort
round apart, and the same request alone on the route already holds them longer than plain decode
when the string crosses a burst boundary (GB10 MTP "Surveyors": 74 ms plain, 180 ms alone on the
route, 530 ms in a cohort of 4 on the fix, 2,498 ms on main). Members that ended on EOS or at
`max_tokens` waited on main too (GB10 MTP: 598 / 699 ms on EOS, 232 / 400 ms at `max_tokens`) and
close at the solo level on the fix (40 / 43 and 39 / 40 ms). Per-row closes: each `runs.jsonl`;
summaries `rtx5090/close-summary.txt`, `gb10/close-summary.txt`.

Judges (`judge-*.log` next to each DIR, `--case` per boot): `fix3` and `fix3-dspark` (RTX 5090)
pass, as do the GB10 final-head boots (`gb10/unit-b-final/`, in DIRs that hold a copy of unit-a's
plain reference rows) and the GB10 `fix` and `fix-dspark` boots; main fails on every device and
route (10 to 31 rows per boot, every one a late close). The first two RTX 5090 MTP fix boots (`fix`, `fix2`) fail two rows
each: they were recorded before the rows carried the cadence the judge's hold-back and EOS
allowance needs (`fix`), or with a first cadence definition that counted chunks inside one round
(`fix2`); their closes are 269 and 237 ms on the hold-back row, 139 and 143 ms on the EOS row.

## Exactness

- Member 0, every row of every boot, main and fix, both devices and routes: text, finish reason and
  completion tokens equal plain decode's solo row (DSpark at temperature 1: the boot's own solo
  row).
- Companions: one text per companion and sampler across every cohort of every boot in the DIR,
  main's included (RTX 5090 MTP: 12 to 92 runs each, DSpark: 9 to 69; GB10 likewise), equal to
  plain decode on MTP and on DSpark greedy. A request's text does not depend on its cohort or on
  the fix.

## Throughput

Four companions sent at once, 192 tokens each; aggregate completion tokens per second from the
first request sent to the last response ended; three cohorts per boot, per-boot median.

RTX 5090, interleaved boots, A/B 1 ABBAAB (97f0364583) and A/B 2 BAABBA (f47c2092a4):

| sampler | fix/main per adjacent pair | median | mean | range |
|---|---|---|---|---|
| t1 | 0.987, 0.989, 1.000, 0.996, 1.001, 0.978 | 0.993 | 0.992 | 0.978 to 1.001 |
| greedy | 1.000, 0.982, 0.988, 1.017, 0.978, 0.987 | 0.987 | 0.992 | 0.978 to 1.017 |

The cohorts inside one boot spread 2 to 4% (`rtx5090/throughput-ab/`), so a difference below about
1% is not resolved by these runs.

GB10 (unit-a), A/B ABBAAB, main ede1b8119f and the fix's code, both built on the unit:

| sampler | fix/main per adjacent pair | mean | range |
|---|---|---|---|
| t1 | 1.001, 1.023, 1.004 | 1.009 | 1.001 to 1.023 |
| greedy | 1.003, 0.999, 1.005 | 1.002 | 0.999 to 1.005 |

On GB10 the cohorts inside one boot spread up to 2% (`gb10/throughput-ab/`).

From the code, the fix adds no per-round device work: the round hook takes one RefCell borrow per
flush (a flag check); the end hook runs once per member per burst, and for a member that continues
it is one `spec_burst_verdict` over the burst with no allocation; only a member whose stream ended
copies its token list (twice, once per request); the engine adds one telemetry snapshot (atomics)
and an O(members) count per member end. In these throughput cohorts every member ends at the same
final round.

## Other gates on the fix

- `tools/seeded-invariance-gate.py`, RTX 5090, 97f0364583: PASS, 54 rows, cases ref, K=0..3 and
  demote (`rtx5090/seeded-invariance-97f0364583/`); GB10, the same code: PASS, 54 rows
  (`gb10/seeded-invariance-97f0364583-code/`).
- `spec-burst-boundary-gate.py` (memra#1186's gate as of its branch; the copy used is
  `tools/spec-burst-boundary-gate.py` here), RTX 5090, 97f0364583, `--carry no` (this binary has
  no sampled pending carry), `--compare` main's run of the same gate: PASS, 229 rows; text, finish
  and completion tokens equal to main's on all 229 (`rtx5090/burst-boundary-97f0364583/`).
- `session-gate` K=2 and K=3, RTX 5090: ALL TURNS MATCH, including the stopped-burst check
  (`rtx5090/session-gate-97f0364583.log`; the binary is byte-identical on the final head).

## CPU tests

On the build server (nvcc 13.1, `MEMRA_CUDA_ARCH=120a`), final head f47c2092a4
(`cpu/green-f47c2092a4/`): `cargo fmt --all -- --check` 0; `cargo test -p memra-server
--no-fail-fast` 0 (lib 1,195 passed); `cargo clippy -p memra-server -p memra-engine --all-targets
-- -D warnings` 0. `cargo test -p memra-engine --lib` on 97f0364583: 738 passed, one failed on a
research fixture the build tree had not received (`argmax_margin` reads
`research/argmax-gate-rule-20261110/`); with the fixture synced it passes.

New tests: `spec_burst_verdict_is_the_post_steps_verdict` (the moved verdict against the post step's
old inline code over a grid of bursts, request rooms, stop-string positions and budgets), `pooled_member_closes_at_its_round_with_the_post_steps_receipt`
(each early receipt equals the post step's computation from the same burst, its events are
`finish`'s; nothing early for a continuing member, a gone client or mismatched sends),
`finish_sends_nothing_after_an_early_done_and_checks_it`, `a_step_error_after_an_early_done_is_not_sent`.
Red (`cpu/red/`): with the early close removed (`red.patch`), the pooled-member test fails; main
itself is the device red arm.

## What the gate learned

- With the spec gate off, DSpark admits one session at a time (`dspark_load_admits`: the gate-off
  arm admits only with no live session), so the first main DSpark boot formed no cohort: in each
  cohort one member speculated and the others ran on plain decode (`rtx5090/dspark-gate-off/`, kept
  as a receipt). DSpark cohort boots run with the gate on and its LOW band at the cohort size.
- Plain decode alone holds a stop string's text back for a decode step or a few; in a cohort the
  same tokens arrive a cohort round apart, and on the route a held string can cross a burst
  boundary. The judge's allowance for a stop-ended member is twice the larger of plain decode's
  hold-back, two of the row's rounds per plain step held back, and the route's own solo hold-back
  scaled by the cohort's cadence over the solo row's; one round on EOS; none at `max_tokens` (the
  independent review asked for the scoping, `review.md`).
- A member 0 that stops in its first, serial burst (before its companions joined the cohort) has no
  cohort to leave; the judge counts those separately and requires, per sampler, at least one member
  0 that finished inside its cohort while another member ran (11 to 19 per sampler on the MTP
  fix boots).

## Files

`rtx5090/` and `gb10/`: one DIR per gate as the gate writes it (`runs.jsonl`, `boots.jsonl`,
`plan.json`, `texts/`, `server-*.log`, `env-*.json`) plus `judge-*.log`. Machine paths in the logs
read `$MODELS` (the model directory) and `$LANE` (this lane's scratch directory); the two GB10 units
are `unit-a` and `unit-b`. `review.md`: the independent review and its dispositions.
