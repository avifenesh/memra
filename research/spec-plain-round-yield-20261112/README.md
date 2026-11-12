# A plain session behind speculative bursts advances once per round (memra#1244)

Date: 2026-11-12. Qwen3.8-27B NVFP4 + Q5_K with its MTP head, every switch at its default.
Decision record: [`docs/decisions/SPEC-ROUND-YIELD.md`](../../docs/decisions/SPEC-ROUND-YIELD.md).
Follows `research/spec-cohort-newcomer-20261112/` (memra#1227), whose head is the parent arm here.

**Verdict: adopted (memra#1244).** A plain request that could not ride a pooled cohort got one
token per speculative burst: 1.36, 1.61 and 1.84 s between tokens beside one, two and three
streams on GB10, where it gets 0.072 s alone. A host-sampled plain session (greedy with a
penalty) now rides the cohort, and while any other plain session waits behind the bursts (a
constrained, vision or fork request, a rider candidate no cohort carries, a newcomer whose
prime completes in one chunk) every MTP burst ends at its first round. GB10: greedy with a
penalty 0.10 to 0.13 s between tokens, the streams 1 to 5% slower while it is live;
json-constrained 0.17 to 0.19 s, the streams 24 to 29% slower while it is live. RTX 5090: 0.06
to 0.08 s where it was 0.46 to 0.72 s, the streams 29 to 41% slower while it is live (this card has no cost table for
the model, so the rider steps plain inside each round). Every text is equal across arms and
equal beside the streams and alone. The seeded-invariance and burst-boundary gates pass.
Not covered: the DFlash2, glm5 and gemma bursts (memra#1246), the full burst a constrained
request waits before admission (memra#1249), a long prime's chunks between full bursts
(memra#1253).

## Conditions

- **GB10**: one DGX Spark unit per comparison, both arms on it, one boot per lease, arms
  interleaved (ABBA or BAAB). `121a` build.
- **RTX 5090 Laptop GPU**: enforced power limit 150 to 160 W. `120a` build.
- **Binaries**:

  | arm | tree | GB10 | RTX 5090 |
  |---|---|---|---|
  | `parent` | 8d8a6ce08c (#1227's head then) | `memra-0.141.1-51f0eb72655f`, sha256 `53ed074a...` | sha256 `8d3d634cc9bb...` |
  | `change` | 00a72fc60b on it | `memra-0.141.1-a1f586cfa120`, sha256 `bbc34b63...` | sha256 `b4262dc49fde...` |
  | `parent2` | 51b933a584, #1227's final head | `memra-0.141.1-2c65e91e03d3`, sha256 `f144447bf402...` | sha256 `ed4791b87bd9...` |
  | `change2` | 7107ce3eba on it (this branch before the rebase onto main) | `memra-0.141.1-6ac6d0891c9c`, sha256 `81e7da7bc66d...` | sha256 `e44719014080...` |

  `mech` and `mech-burst1` ran main 340a5e5bfa (the same crates as #1241's head a20fd8c59f).

## Cells

`tools/plain_cell.py`: L long speculating streams start, then after 8 s plain requests of
NEWTOK tokens go one after another beside them, then the same prompts alone. A row per boot and
cell in `plain.jsonl`: every token gap of each plain request, its TTFT and E2E, the streams'
rate while a plain request is live and while none is, and the texts' hashes (texts under
`texts/`). `tools/summ_plain.py DIR...` prints the tables and the text checks (exit 0 = texts
identical and stream texts prefix-equal). Server logs are gzipped when large.

| dir | what | arms, order |
|---|---|---|
| `gb10/mech` | main, the mechanism: pen plain beside L = 1, 2, 3 greedy streams, tick trace | base |
| `gb10/mech-burst1` | main with `MEMRA_SPEC_BURST=1` (rejected: no round runs) | burst1 |
| `gb10/plain` | plain cell, L = 1, 2, 3, greedy and t1 streams, pen and t1json plain | parent, change, change, parent |
| `gb10/t1yield` | sampled (t1) streams beside a t1json plain request | parent, change, change, parent |
| `gb10/new` | newcomer cell (`tools/newcomer_cell.py`, memra#1227's cell), L = 0 to 3 | change, parent, parent, change |
| `gb10/longprime` | a 23,444-token pen prompt beside 2 greedy streams | parent, change, change, parent; parent2, change2, change2, parent2 |
| `gb10/plain2` | plain cell on the final pair | change2, parent2, parent2, change2 |
| `gb10/new2` | newcomer cell on the final pair | parent2, change2 |
| `gb10/inv-change`, `gb10/inv-change2` | `tools/seeded-invariance-gate.py` on change, change2 | |
| `gb10/bnd-change`, `gb10/bnd-change2` | `tools/spec-burst-boundary-gate.py` on change, change2 | |
| `rtx5090/plain` | plain cell, L = 1, 2 | parent, change, change, parent |
| `rtx5090/plain2` | plain cell on the final pair, one boot per arm | parent2, change2 |
| `rtx5090/new` | newcomer cell | parent, change, change, parent |
| `rtx5090/norm` | closed loop c = 1, 2, 4, 8 greedy (`normal_cell.py`), one boot per arm | parent, change |
| `rtx5090/inv-change2` | seeded-invariance gate on change2 | |
| `cpu/` | CPU checks on bbcec42e31, the rebased head (fmt, clippy, engine and server suites, flags and numeric censuses), and the red arms | |

Void: `gb10/longprime/void.jsonl` and `void-server-parent2-4.log.gz` are the parent2 boot 4
whose lease overlapped another lane's page-cache fill on the unit; boot `parent2-5` replaces it.

## Results

Every table of the record comes from `SUMMARY.md`, which is the output of the two summary
tools over these dirs (each command is in its heading). Per dir:

- `gb10/plain`, `gb10/plain2`: greedy with a penalty 1.34 to 1.84 s -> 0.100 to 0.132 s gap p50
  at L = 1 to 3, streams' rate while it is live 1 to 5% lower (one row 5% higher); json-constrained 1.36 to
  1.73 s -> 0.172 to 0.188 s, streams -24 to -29% while it is live. Both pairs agree within
  0.002 s on every change gap. 144 plain texts identical per pair, stream texts prefix-equal.
- `gb10/t1yield`: t1 streams beside a t1 json request: 1.447/1.728 -> 0.173/0.188 s, streams
  22.4/38.4 -> 15.8/28.9 tok/s while it is live.
- `rtx5090/plain`, `rtx5090/plain2`: greedy with a penalty 0.46/0.72 -> 0.059/0.073 to 0.076 s,
  streams -41% and -32% while it is live (the rider steps plain inside each round on this
  card); json-constrained 0.47 -> 0.060 s, streams -29%.
- `gb10/new`, `gb10/new2`: first-token time behind a cohort unchanged (p50 within 0.013 s,
  p95 within 0.005 s), long rates equal. `rtx5090/new`: the third request at L = 2, which the
  band admits plain, rides: 0.842 -> 0.202 s (greedy), 0.896 -> 0.214 s (t1) p50.
- `rtx5090/norm`: aggregate 73.2/94.7/116.4/146.5 -> 73.5/95.4/118.3/154.0 tok/s at c = 1, 2, 4,
  8 (one boot per arm, no plain request).
- `gb10/longprime`: TTFT 57.6 to 59.8 s -> 55.0 to 55.9 s (19.4 to 19.6 s alone), decode gap
  1.59 -> 0.119 s; the chunks keep their full bursts (memra#1253).
- `gb10/mech`: on main, 13 rounds per burst of 2.54 tokens, 99/111/126 ms a round beside 1/2/3
  streams; the plain request's gap p50 1.37/1.62/1.84 s against 0.074 s alone.
- `gb10/mech-burst1`: `MEMRA_SPEC_BURST=1` runs no speculative round: streams 6.6/8.8/9.9 tok/s,
  plain gap 0.15/0.23/0.30 s. Rejected.
- Gates: seeded invariance PASS (54 rows) on `gb10/inv-change`, `gb10/inv-change2` and
  `rtx5090/inv-change2`; burst boundary PASS (229 rows) on `gb10/bnd-change` and
  `gb10/bnd-change2`. Each verdict is the `PASS` line of the dir's `gate.log`.

## Red arms

`tools/redarm.py` writes each arm into a copy of the tree, `tools/dd-redgreen.sh` runs the
three new tests on each (`cpu/redgreen.log`): `parent` (the change reverted) and `nohook` (no
round hook) fail the burst and wiring tests and pass the naming test; `nodue` fails the naming
test; `nowrite` and `noride` fail the wiring test; the change passes all three.
