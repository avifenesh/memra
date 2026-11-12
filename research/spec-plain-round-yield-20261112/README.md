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
- **RTX 5090 Laptop GPU**: `120a` build. Enforced power limit 150 to 160 W for the first
  cells. The review-head cells (`riders5`, `riders6`, `riders7`, `plain5`, `new5`, `inv-change5`) ran
  after a reboot, on AC: 150 to 160 W at a boot's start and 175 W under load (a 150 W TGP plus
  25 W of dynamic boost), recorded per boot in `guard.log` and not changed.
- **Binaries**:

  | arm | tree | GB10 | RTX 5090 |
  |---|---|---|---|
  | `parent` | 8d8a6ce08c (#1227's head then) | `memra-0.141.1-51f0eb72655f`, sha256 `53ed074a...` | sha256 `8d3d634cc9bb...` |
  | `change` | 00a72fc60b on it | `memra-0.141.1-a1f586cfa120`, sha256 `bbc34b63...` | sha256 `b4262dc49fde...` |
  | `parent2` | 51b933a584, #1227's final head | `memra-0.141.1-2c65e91e03d3`, sha256 `f144447bf402...` | sha256 `ed4791b87bd9...` |
  | `change2` | 7107ce3eba on it (this branch before the rebase onto main) | `memra-0.141.1-6ac6d0891c9c`, sha256 `81e7da7bc66d...` | sha256 `e44719014080...` |
  | `parent3` | cdfdfd6006, main (#1227 and #1233 landed) | `memra-0.141.1-14a59677bb0e`, sha256 `00b8744f648f...` | sha256 `70130b7d8202...` |
  | `change3` | 9074ec16aa, the first review head (superseded) | `memra-0.141.1-a88473c260ed`, sha256 `48caeb28da7e...` | sha256 `fa3e3999b2d6...` |
  | `change5` | 3eeb4710f0, the review head on cdfdfd6006 (rebased onto main 11b368f7d7 as 29b03f85fb with the same delta; 76af934425 adds a test line only) | `memra-0.141.1-1f4dcebc4fe7`, sha256 `a43e7ff33839...` | sha256 `33af2f820d97...` |
  | `parent4` | 11b368f7d7, main (memra#1248 and memra#1259 landed) | `memra-0.141.1-32737cebc309`, sha256 `9ee0eb49ada1...` | |
  | `change6` | 11ae9661d1, the round-2 review head on it | `memra-0.141.1-a18ac57c28ec`, sha256 `66e3098a72ec...` | sha256 `e73828fc3cd9...` |

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
| `gb10/riders` | riders cell (`tools/riders_cell.py`): 2 greedy streams, then 4 greedy-with-penalty riders, 4 and 8 of the vendor shape (`MEMRA_SERVE_DEVPENALTY=0`), one boot per arm | parent3, change3 |
| `gb10/plain3` | plain cell with a re-send, 40-token prompts (under the cache minimum: no hit), the first review head; shows the outlook regression the time-window rule fixed | parent3, change3 |
| `gb10/inv-change3`, `gb10/bnd-change3` | both gates on change3 (superseded head) | |
| `gb10/plain5` | plain cell, 1,234-token prompts and a prefix-hit re-send, review head | parent3, change5 |
| `gb10/riders9` | riders cell, 8 and 6 greedy-with-penalty riders, review head | change5, parent3 |
| `gb10/inv-change5`, `gb10/bnd-change5` | both gates on the review head | |
| `rtx5090/riders5` | riders cell, 4 and 8 greedy-with-penalty riders, 4 and 8 of the vendor shape (`MEMRA_SERVE_DEVPENALTY=0`) | parent3, change5 |
| `rtx5090/riders6` | riders cell, 2 streams + 2 riders and 1 + 3 (4 in flight) | change5, parent3 |
| `rtx5090/riders7` | riders cell, 1 stream + 2 riders and 2 + 1 (3 in flight) | parent3, change5 |
| `rtx5090/plain5` | plain cell, 1,234-token prompts and a prefix-hit re-send, L = 1, 2 | change5, parent3 |
| `rtx5090/new5` | newcomer cell, L = 1, 2 | parent3, change5 |
| `rtx5090/inv-change5` | seeded-invariance gate on the review head | |
| `gb10/plain6` | plain cell, 1,234-token prompts and a prefix-hit re-send, round-2 head | parent4, change6 |
| `gb10/mixed` | `tools/mixed_cell.py`: an interactive greedy-with-penalty rider and an interactive json-constrained request a second later (`2:pen~0+t1json~1`); an interactive json request without (`2:t1json~3`) and with (`2:pen@harvest~0#48+t1json~3`) a dark harvest request, beside 2 greedy streams | change6, parent4 |
| `gb10/inv-change6`, `gb10/bnd-change6` | both gates on the round-2 head | |
| `gb10/mixed-shed` | the first change6 boot of the mixed cell: the harvest request shed with a 429 (kept as the receipt of the shed) | change6 |
| `rtx5090/inv-change6` | seeded-invariance gate on the round-2 head | |
| `cpu/` | CPU checks on 11ae9661d1, the round-2 head on main 11b368f7d7 (fmt, clippy, engine and server suites, flags and numeric censuses), and the red arms | |

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
  band admits plain, rides in both arms (first to second token 0.035 and 0.034 s); `Priming`
  ends the tick's bursts before its prime, so its first token comes 0.842 -> 0.202 s (greedy),
  0.896 -> 0.214 s (t1) p50, and the streams give up 3% (greedy) and 8% (t1) for it.
- `rtx5090/norm`: aggregate 73.2/94.7/116.4/146.5 -> 73.5/95.4/118.3/154.0 tok/s at c = 1, 2, 4,
  8 (one boot per arm, no plain request).
- `gb10/longprime`: TTFT 57.6 to 59.8 s -> 55.0 to 55.9 s (19.4 to 19.6 s alone), decode gap
  1.59 -> 0.119 s; the chunks keep their full bursts (memra#1253).
- `gb10/mech`: on main, 13 rounds per burst of 2.54 tokens, 99/111/126 ms a round beside 1/2/3
  streams; the plain request's gap p50 1.37/1.62/1.84 s against 0.074 s alone.
- `gb10/mech-burst1`: `MEMRA_SPEC_BURST=1` runs no speculative round: streams 6.6/8.8/9.9 tok/s,
  plain gap 0.15/0.23/0.30 s. Rejected.
- Review head (`change5` against main `parent3`, one boot per arm):
  - `gb10/plain5`, `rtx5090/plain5` (1,234-token prompts): greedy with a penalty 1.608 ->
    0.115 s gap p50 on GB10 (L = 2), 0.500/0.746 -> 0.054/0.071 s on the 5090 (L = 1, 2);
    the prefix-hit re-send's first token 1.57/1.61 -> 0.07 to 0.13 s (GB10) and 0.86/0.82
    -> 0.19 to 0.23 s (5090). The total while the request is live falls on every row of these
    two dirs (a latency-for-throughput trade). Texts identical.
  - `gb10/riders10`, `gb10/riders9`: 4 greedy-with-penalty riders beside 2 streams ride on
    GB10: gap 1.607 -> 0.132 s, streams -17% while they are live, total +46%. With 5, 6 and 8
    riders the cost model demotes the streams to batched decode in both arms, and the arms are
    equal. `gb10/riders` ran the 4-rider cell on the first review head (1.615 -> 0.133 s).
  - `rtx5090/riders5`: 4 and 8 riders put 6 and 10 sessions in flight, above the band's
    HIGH = 4 on this card, so the streams are demoted in both arms and the arms are equal.
    `rtx5090/riders6`: 2 + 2 and 1 + 3 (4 in flight) are demoted too, both arms.
    `rtx5090/riders7` (3 in flight, the only shape where several riders ride on this card): 1
    stream + 2 riders 0.486 -> 0.057 s gap, the stream -37% while they are live, total +6%; 2
    streams + 1 rider 0.751 -> 0.069 s, streams -27%, total -12%.
  - `rtx5090/new5`: the third request at L = 2, admitted plain and riding in both arms, gets
    its first token 0.860 -> 0.212 s (greedy), 0.999 -> 0.207 s (t1); L = 1 0.159 -> 0.170 s.
    Newcomer texts prefix-equal.
  - `gb10/plain3` (the first review head, `change3`): a newcomer's first token 1.79 s where
    `change2` had 0.31 s, because the prime outlook read phase (b)'s 4 ms windows at the tick's
    start. The time-window rule of the review head answers it.
- Round-2 head (`change6` against main `parent4`, one boot per arm):
  - `gb10/plain6`: repeats `gb10/plain5` within 0.003 s on every gap; texts identical.
  - `gb10/mixed`: an interactive penalty rider kept in phase (c) by an interactive json request
    (the `rider-uncarried` receipt) 1.607 -> 0.115 s, the json request 1.597 -> 0.189 s; a dark
    harvest request beside the json request keeps one token per burst's worth of time
    (1.610 -> 1.619 s) and costs the streams 2% (30.1 -> 29.5 tok/s; 1% on main). The harvest
    request is admitted with `MEMRA_SHED_HARVEST=100`; `gb10/mixed-shed` is the first change
    boot, where the default shed it (429) beside the streams and alone.
- Gates: seeded invariance PASS (54 rows) on `gb10/inv-change`, `gb10/inv-change2`,
  `gb10/inv-change3`, `gb10/inv-change5`, `gb10/inv-change6`, `rtx5090/inv-change2`,
  `rtx5090/inv-change5` and `rtx5090/inv-change6`; burst boundary PASS (229 rows) on
  `gb10/bnd-change`, `gb10/bnd-change2`, `gb10/bnd-change3`, `gb10/bnd-change5` and
  `gb10/bnd-change6`. Each verdict is the `PASS` line of the dir's `gate.log`.

## Red arms

`tools/redarm.py` writes each arm into a copy of the tree, `tools/dd-redgreen.sh` runs the seven
tests on each (`cpu/redgreen.log`, on the round-2 head 11ae9661d1):
- `parent` (the change reverted) and `nohook` (no round hook) fail the burst and wiring tests;
- `nodue` (the verdict never says yes) and `nomtp` (it ignores the tick's bursts) fail the
  naming test;
- `nowrite`, `noride` (no host rider), `noridelane` (host riders in every lane), `nolanewire`
  (the tick weighs every live plain session), `nooutlook` (no phase (b) hold), `carriedB` (no
  cohort candidate in the riding rule, the second review's arm B) and `nocadencewire` (phase (c)
  and (d) ignore the cadence) fail the wiring test;
- `nolane` (the lane helper ignores the lane) fails the lane test;
- `noheld` (the prime helper ignores the hold), `nowindows` (the time windows never close) and
  `nochoice` (a waiting choice group or the walker primes) fail the prime test;
- `carriedA` (rider candidates hold riders back, the second review's arm A) and `carriedC`
  (dark decoders hold riders back) fail the carried test;
- `nocadence` (dark work on every yielded tick) fails the cadence test.

The change passes all seven.
