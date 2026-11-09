# prefix-budget-20261107: the discrete-card prefix cache (budget, victim order, admission reclaim)

Lane `fix/discrete-prefix-budget` (memra#1090), base main `6f5e03e74a`. Decision record:
[`docs/decisions/PREFIX-CACHE-DISCRETE.md`](../../docs/decisions/PREFIX-CACHE-DISCRETE.md).

Verdict: the discrete budget leaves the session peak (on one RTX PRO
6000 Max-Q at `MEMRA_MAX_SESSIONS=12` it takes the budget from 2.26 GB to 54.3 GB, repeats hit 30/30 against
0/30, agentic c1 and c4 cache 99.5% of prompt tokens against 19.8% and 16.8%, no OOM; with the cache at its limit
one defer flushes the whole 54.2 GB, with no OOM and pressure TTFT within about 0.15 s of main; on the 5090 it is a
no-op for Qwen3.8-27B at the measured caps). Value-per-byte eviction won the mixed shape (6/6 repeats against 3/6) and tied the
2026-09-21 incident replay 10/10, but lost the agentic c1 cell's shared 256-token head (16 against 24 hits,
-5.7% cached tokens): rejected. The partial admission reclaim was flat with plain LRU and the live-free insert
clamp lost the shared entry: both rejected. The rejected levers' code is banked on
`bank/prefix-budget-gds-20261107`.
Model in every cell: Qwen3.8-27B NVFP4 + Q5_K GGUF with its MTP head (`Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`),
`MEMRA_CTX=32768` unless stated. Card for every scored cell outside the RTX PRO 6000 sections: one RTX 5090
Laptop GPU (24 GB), balanced platform profile, 74-80 C under load. The PRO 6000 sections name their card. Local absolute paths in the receipts are replaced by
placeholders (`<lane>`, `<models>`, ...).

## The finding (one RTX PRO 6000 Max-Q, 96 GB): `pro6000-finding/`

The spec-cache-mixed gate (`tools/spec-cache-mixed-gate.sh`: one greedy warm request with a 520-token
shared prompt, then c8 unique-prompt batch load with six greedy repeats of the shared prompt 3 s apart,
`MEMRA_MAX_SESSIONS=12`):

- boot line: `budget 2260MB (2259943424 B, derived: 2 x 1129971712 B max entry ... boot driver free
  86614081536 B, post-reserve clamp 85003468800 B)`;
- a 64-token seed is 158.8 MB, the shared 480-token spec-boundary entry 172.1 MB: every entry carries
  156.9 MB of GDN recurrent state, against 29.7 KB per cached token;
- 14 seeds fill the budget; the shared entry was evicted before the first repeat:
  `cached=[0, 0, 0, 0, 0, 0]`, 105 eviction lines.

The 24 GB RTX 5090 Laptop is memory-bound for this model: boot free 10,053,877,760 B at
`MEMRA_MAX_SESSIONS=12` (8,342,601,728 B at the default caps), and the session peak of every measured
configuration exceeds it, so the derived budget is the two-entry floor (2,259,943,424 B) in every 5090
cell. The budget arm changes nothing on this card; its effect is the PRO 6000's (measured below).

## The owed run, measured (one RTX PRO 6000 Max-Q, 2026-11-08): `pro6000-maxq/`

One RTX PRO 6000 Blackwell Max-Q Workstation Edition (96 GB) per comparison, 280 W limit (the card's set limit; no
raise, so each A/B kept one power regime), no thermal throttle, Qwen3.8-27B NVFP4 + Q5_K with its MTP head,
`MEMRA_CTX=32768`. A = main `2288c4bf82`, B = this change on it (`19316c0049`; the code diff is byte-identical to
the landed one). Every cell is a fresh boot; arms interleave A B B A A B on one card. The workloads are the 5090
lane's: `harness/mixed.py` (the spec-cache-mixed shape) and `harness/agentic.py` (29 turns of 18 coding-agent
sessions, prime then measure, T=1, 768 new tokens).

| cell | arm | budget (boot line) | result | device free floor | OOM / reclaim-on-defer |
|---|---|---|---|---|---|
| mixed, `MEMRA_MAX_SESSIONS=12`, 5 runs per arm | A | 2,259,943,424 B | repeats hit 0/30, 121 evictions per run | 70.8 GiB | 0 / 0 |
| | B | 54,277,046,272 B (24 sessions x 1,190,789,120 B) | repeats hit 30/30 (480 cached tokens each), 0 evictions | 55.8 GiB | 0 / 0 |
| mixed, default caps, 3 runs per arm | A, B | 2,259,943,424 B on both (the floor) | 0/18 hits on both, 113-121 evictions | 62.8 GiB | 0 / 0 |
| agentic c1, `MEMRA_MAX_SESSIONS=12`, 3 runs per arm | A | 2,260 MB | 19.8% cached (36,448 of 184,051 tokens), 53 evictions | 70.8 GiB | 0 / 0 |
| | B | 54,277 MB | 99.5% cached (183,200), 0 evictions | 64.8 GiB | 0 / 0 |
| agentic c4, same | A | 2,260 MB | 16.8% cached (30,912), 56 evictions | 69.0 GiB | 0 / 0 |
| | B | 54,277 MB | 99.5% cached (183,200), 0 evictions | 60.7 GiB | 0 / 0 |

Timing, same cells (single-window medians; see the conditions note below):

| cell | A | B |
|---|---|---|
| mixed MS=12, repeat TTFT median over 30 repeats | 327.3 ms (mean 466.4) | 193.1 ms (mean 362.6) |
| mixed MS=12, batch tok/s (median of runs) | 224.6 | 225.0 |
| mixed MS=12, batch TTFT p50 (median of runs) | 567.1 ms (runs 489-609) | 596.1 ms (runs 576-600) |
| agentic c1, TTFT p50 / p95 | 1,103-1,107 / ~6,250 ms | 108-111 / 134-137 ms |
| agentic c1, aggregate tok/s | 68.4-68.6 | 105.6-106.8 |
| agentic c4, TTFT p50 / p95 | 1,750-1,763 / ~7,000 ms | 438-677 / 802-853 ms |
| agentic c4, aggregate tok/s | 91.1-91.7 | 155.6-159.1 |

The mixed shape's repeat TTFT is bimodal on both arms: a repeat that lands behind a c8 batch wave waits 560-900 ms,
on A and on B and in the default-caps phase where both arms are the same code path, so the tail is queueing, not
the cache. A hit that is not queued answers in 175-197 ms against 297-380 ms cold.

Exactness: on the mixed shape every compared greedy row (repeats and batch rows) is byte-identical to the
cache-off run, in every arm and run. Agentic greedy (256 tokens, `MEMRA_MAX_SESSIONS=12`): B with 29/29 hits gives
29/29 turns byte-identical to a cache-off boot; A 29/29 as well.

**Conditions, stated so these timings are not read as quiet-box numbers.** The box has four cards on one PCIe
complex, and other work ran on its other cards during these cells: a timed A/B record on another card during the
first mixed phase and the default-caps phase, a two-card correctness campaign on two other cards during the
agentic cells, and the agentic c1 and c4 phases ran at the same time on two different cards. Each comparison's arms
ran interleaved on one card in one window, so A against B stays fair, but the absolute timings carry that load.
The cache counts, budgets, memory floors and byte identity do not depend on it. A quiet-box re-time of the
agentic headline cells is owed before any of these timings is published.

Scripts: `pro6000-maxq/campaign/` (`arm.sh` is the 5090 lane's arm pinned to one card by UUID; `phases-1090*.sh` order the arms; `m12_table.py` makes the mixed table). Each run dir keeps `arm.txt` (binary hash, env, boot budget line, device free and MemAvailable floors, eviction, OOM and reclaim counts), `serve.log`, the 250 ms `gpu-tel.csv`, `memavail.txt`, the result and the rows (agentic rows as numbers plus output sha256).

## The cache at its limit (one RTX PRO 6000 Max-Q, 2026-11-09): `pro6000-maxq/limit/`

The owed cells above never filled the cache: it peaked at 20.5 GB (mixed) and 11.2 GB (agentic) of the 54.3 GB
budget, with no reclaim. This cell fills it. `harness/limit.py`, `MEMRA_CTX=32768`, `MEMRA_MAX_SESSIONS=12`, every
request greedy, streamed and in its own `cache_salt` namespace (so finished sessions park, up to 16):

- fill: 64 unique prompts of about 26.5k tokens, four at a time; each leaves one prefix entry (about 0.95 GB) and parks its
  session;
- pressure: 12 more unique prompts of about 25.5k tokens at once (the interactive cap);
- repeat: the 16 most recent fill prompts again, one at a time.

A = main `9877f6a696`, B = this change on it (`aabb3ae1b6`; the review fixes after it change comments, tests and
a pure-function split of the session count, not behavior), cell_evidence's own release builds of both, four fresh boots interleaved A B B A on wrx GPU 3 at its 280 W
limit, no thermal throttle. Table: `harness/limit_table.py`.

| arm | budget MB | reclaim-on-defer flushes | entries / GB flushed | evictions preflight / LRU | OOM lines | pressure: errors, TTFT p50 / max s | repeats: hits, TTFT p50 s |
|---|---|---|---|---|---|---|---|
| limit-1-A | 2260 | 0 | 0 / 0.0 | 75 / 11 | 0 | 4 of 12, 53.3 / 88.7 | 0 of 16, 10.61 |
| limit-2-B | 54277 | 1 | 57 / 54.2 | 7 / 0 | 0 | 4 of 12, 53.3 / 88.7 | 0 of 16, 10.61 |
| limit-3-B | 54277 | 1 | 57 / 54.2 | 7 / 0 | 0 | 4 of 12, 53.3 / 88.7 | 0 of 16, 10.60 |
| limit-4-A | 2260 | 0 | 0 / 0.0 | 75 / 11 | 0 | 4 of 12, 53.2 / 88.6 | 0 of 16, 10.59 |

What it shows:

- B filled the budget (seven single-entry evictions at the top), then one pressure arrival deferred and
  `reclaim-on-defer` flushed the whole cache: 57 entries, 54.2 GB, effective free 4.6 GB to 58.8 GB. Admission
  charged that arrival 3.4 GB (KV, a 2.3 GB prefill workspace and fixed state); the shortfall was under 1 GB. The
  parked sessions were not touched. No other eviction kind appeared: no post-insert yield and no allocation-failure
  eviction on any arm (the only lines are `evict (LRU)` and `evict (snapshot preflight, LRU)`).
- Against main nothing got worse: no OOM on either arm, pressure TTFT at most about 0.15 s behind main at every rank
  (all 12 arrivals are admitted and the flush runs before the first prefill finishes, so the offset starts at rank 1;
  the arrivals then wait for each other's 25k-token prefill), and the 16 repeats re-prime
  on both arms (0 hits, 10.6 s each). A never held those entries; B held them until the flush.
- Four pressure requests on every arm end with HTTP 408 at the 90 s first-token deadline: they were admitted but
  waited for their prefill turn behind 25k-token prefills, on both arms alike. `admit_oom_lines` in `arm.txt` counts
  the `[admit-oom]` boot banner and the reclaim lines, not out-of-memory events. The harness's Prometheus capture
  read nothing (`/metrics` answers JSON without an `Accept` header), so the table comes from the serve.log lines.
- The cost of the flush is the cache, not latency: a partial reclaim that frees only the arrival's shortfall,
  oldest first, would have kept the repeats' entries. That lever is owed in the decision record, with this cell as
  its acceptance test.

Conditions: the box's other cards ran another lane's training jobs during these arms (about 28 GB each, full
utilization). The decision metrics here are counts (flushes, evictions, hits, OOM lines); the TTFT comparison is
interleaved on one card and both arms carry the same load.

## Push-gate cell records for the landing commit

`research/release-qualification/cells/6a9db1447dbd-rtxpro6000-battery-perf-serving/` and
`.../6a9db1447dbd-gb10-battery-perf-serving/`: battery, perf A/B against main `0c25263389` and serve-smoke, for
`6a9db1447d` (this change with both review rounds' fixes, on that main), on one RTX PRO 6000 Max-Q (sm_120a, wrx
GPU 1) and one DGX Spark GB10 (sm_121a). Both pass.

The rtxpro6000 record ran with the box's quiet flag set (2026-11-09, 06:36:45 to 06:45:17 Jerusalem) and no other
compute on any card. Its phases: battery 06:36:56 to 06:40:18, serve-smoke 06:40:18 to 06:43:17, perf 06:43:17 to
06:44:59. GPU 1's set limit is 250 W; power-cell ran the record at 280 W (the regime of the other PRO 6000 records)
and restored 250 W. One telemetry sample, at 06:40:06, read 250 W after an outside reset and power-cell re-asserted
280 W, so the timed perf phase ran entirely at 280 W; max 77 C, no throttle. Earlier records for `aabb3ae1b6` and
`9e4dd573cf` also passed and are superseded by the review fixes and the rebase.

## Harness (`harness/`)

- `mixed.py`: the spec-cache-mixed workload with every request streamed (TTFT per row, greedy text per
  row, batch rows tagged `batch-<worker>-<item>` so texts compare across arms).
- `agentic.py`: the `ab_agent.py run` shape (prime every turn with `max_tokens=1`, then the measured pass)
  with per-turn TTFT and cached tokens. The workload replays 29 turns of 18 private coding-agent sessions;
  the committed rows keep the numbers and a sha256 of each output (`sanitize_agentic.py`), not the texts.
- `arm.sh`: one boot, one workload, 250 ms `nvidia-smi` telemetry (device free floor) and 1 s
  `/proc/meminfo` MemAvailable; `phase*.sh` interleave the arms, each phase under its own gpulock lease.
- `policy_sim.py`: the trace-free policy simulator below; `prefix-evict-ab.py`: the incident replay.
- `limit.py`: the at-the-limit workload (fill, pressure, repeat; one `cache_salt` namespace per prompt);
  `limit_table.py`: its table from the arms' serve.log and rows.

## Policy simulator: `sim/policy-sim.txt`

Entry bytes 156.9e6 + 29,700 x tokens, the 32-token grid, whole-entry restores, no session pressure.
Arms: lru, gds (always admit, shipped), gds-refuse (admits only by displacing lower priority), door
(second sight of the 64-token family). At 2,260 MB unless stated:

| shape | lru | gds | gds-refuse | door |
|---|---|---|---|---|
| mixed (14 seeds per repeat), repeats hit | 0/6 | 6/6 | 6/6 | 5/6 |
| incident replay (5090 day-20 geometry, 1 GiB), computed tokens | 28,896 | 28,896 | 30,528 | 45,376 |
| agentic prime then measure (29 turns), computed tokens | 261,534 | 261,534 | 261,534 | 286,558 |
| the same at 8 GB | 261,534 | 163,806 | 163,806 | 274,366 |
| mixed and agentic at the PRO 6000's 54,277 MB | 6/6, 29/29 | 6/6, 29/29 | - | 5/6, 11/29 |

## The incident replay (scored): `rtx5090/phase1d/`

`tools/prefix-policy-ab.py` (deleted with the SLRU door, recovered from `9466b891` as
`harness/prefix-evict-ab.py`, arms renamed lru / gds), the day-20 geometry of
`docs/decisions/PREFIX-CACHE-POLICY.md`: budget 1024 MiB, ctx 16384, cohort 1248/1344/1440/1536 ids
seeded twice, one loop 10,912 + 160 ids per turn over 12 turns, returns every third turn,
`MEMRA_REUSE_POOL=0`, AB/BA x 5 pairs, one lock hold, a cache-off boot for the cold digests. Binary
`7b385110e4` (its victim-order code is the shipped code; run with `MEMRA_ENV_AUDIT=warn` because its door
registry predates the `MEMRA_PREFIX_EVICT` row; the first attempt refused on exactly that and is kept in
`attempt1-env-audit-refused/`). Verdict, verbatim:

```text
PREFIX-POLICY-AB: budget_bytes=1073741824 cohort_tenants=4 cohort_bytes=792920064 turns=12 start_tokens=10912 grow=160 return_every=3 pairs_per_order=5 runs=20 requests_per_run=28 digests_identical=28/28 computed_tokens lru_median=29600 gds_median=29600 (N=10 each) pairs_lru_better=0/10 pairs_gds_better=0/10 ties=10/10 return_cached lru_median=1696 gds_median=1696 loop_cold_after_1 lru_max=0 gds_max=0 refusals lru=0 gds=0 temp_c=78.0..80.0 power_limit_w=None -> INCONCLUSIVE
```

Ten ties of ten: both orders computed 29,600 tokens per run (the day-20 LRU median, to the token),
returned 1,696 cached tokens to the cohort and matched the cold digests 28/28. The only difference is one
more eviction per run under gds (20 vs 19). Loop TTFT p50 188-194 ms in both arms. Value per byte does not
lose the shape the 2026-09-21 decision was made on.

## The live-free insert clamp, measured and rejected: `rtx5090/phase1a-clamp/`, `rtx5090/preview1a/`

The first scored candidate (`39ec446dd5`) also applied GB10's live-free insert clamp on discrete cards. The
mixed shape, three runs per arm interleaved ABCDE EDCBA ABCDE, `MEMRA_MAX_SESSIONS=12`:

| arm | repeats hit (each run) | repeat TTFT median ms | batch tok/s | batch TTFT p50 ms | reclaims | device free min MiB |
|---|---|---|---|---|---|---|
| base (`6f5e03e74a`) | 2, 3, 3 | 489 | 94.9 | 1471 | 6 | 1267 |
| clamp + partial reclaim, LRU | 0, 0, 0 | 763 | 93.1 | 1456 | 0 | 2630 |
| clamp + partial reclaim, GDS | 1, 1, 1 | 758 | 96.2 | 1489 | 0 | 2611 |
| clamp + reclaim flush, GDS | 1, 1, 1 | 797 | 94.4 | 1521 | 0 | 2611 |
| no clamp + partial reclaim, GDS (`7b385110e4`) | 6, 6, 6 | 350 | 95.5 | 1438 | 18 | 1235 |
| off (one run) | 0 | 664 | 110.2 | 688 | 0 | 2931 |

Under load the clamp held the cache to about one entry, and since an insert never evicts itself, every
new seed displaced the shared entry. Without the clamp the value-per-byte order kept it through every
repeat. Greedy texts matched the cache-off run on every compared row in every arm. The clamp is not in
the shipped change; the preview (`preview1a/`, single runs of `7b385110e4`) had shown the same 6/6.

The agentic c8 preview (`preview1b/`) answered 408 on both base and `7b385110e4`: 29 prompts of up to
15k tokens at c8 do not reach a first token inside the 90 s streaming deadline on 24 GB, so the scored c8
cell runs with `MEMRA_STREAM_TTFT_MS_MAX=900000` as a reclaim stress cell, compared only across arms.

## Scored arms, final candidate (`cand-c003e4eded`, no clamp): `rtx5090/phase1a-final/`, `phase1b/`, `phase1c/`

Arms: base = main `6f5e03e74a`; lru = the candidate with `MEMRA_PREFIX_EVICT=lru` (session-peak budget, which is
the floor here, plus the partial reclaim with the owed-bytes term); gds = the candidate's default (+ value per
byte); flush = a measurement-only build of the candidate with main's reclaim flush (value per byte alone).

The mixed shape, three runs per arm, ABCD DCBA ABCD, `MEMRA_MAX_SESSIONS=12`:

| arm | repeats hit (each run) | repeat TTFT median ms | batch tok/s | batch TTFT p50 ms | reclaims | device free min MiB | greedy == off |
|---|---|---|---|---|---|---|---|
| base | 3, 3, 3 | 656 | 96.0 | 1452 | 6 | 1235 | 91/91, 88/88, 87/87 |
| lru | 2, 2, 3 | 689 | 98.0 | 1347 | 15 | 1267 | 95/95, 95/95, 89/89 |
| gds | 6, 6, 6 | 355 | 96.6 | 1456 | 11 | 1235 | 87/87, 87/87, 87/87 |
| flush | 4, 4, 5 | 431 | 95.3 | 1545 | 6 | 1235 | 87/87, 87/87, 87/87 |
| off (one run) | 0 | 672 | 110.7 | 690 | 0 | 3091 | 95/95 |

Per run, the repeats' cached tokens: base `[480, 0, 0, 512, 0, 512]` x3; gds `[480]` x6 in every run; flush loses
the first two or three repeats after a flush and then hits the 512-token seed a missed repeat publishes. Greedy
texts are byte-identical to the cache-off run on every compared row (warm, repeats and every batch row whose
`batch-<worker>-<item>` tag both runs served). Host MemAvailable never fell below 29 GiB (it is not the binding
pool on a discrete card); the device free floor was 1.2-1.3 GiB in every arm with a cache.

The agentic cell (29 turns of 18 coding-agent sessions, prime then measure, T=1, 768 new tokens, default caps),
two runs per arm AB BA:

| cell | arm | measured hits | cached fraction | TTFT p50 ms | TTFT p95 ms | agg gen tok/s | reclaims |
|---|---|---|---|---|---|---|---|
| c4 | base | 14, 10 | 0.164 | 3964 | 19789 | 42.41 | 36 |
| c4 | gds | 13, 14 | 0.165 | 4013 | 18556 | 41.61 | 36 |
| c1 | base | 24, 24 | 0.194 | 3263 | 18569 | 26.95 | 13 |
| c1 | gds | 16, 16 | 0.182 | 3310 | 18464 | 26.96 | 34 |

At c1 the sessions share a 256-token spec-boundary head (165.0 MB, almost all recurrent state): base hit it 12
times per run, gds 4, because its value per byte (1.6 tokens per MB) sits far below a long turn's (about 25), so
every insert of a dense turn passes it. Cached tokens per run: base 35,616, gds 33,568 (both runs identical).
`sim/policy-sim-shared-head.txt` reproduces it offline (LRU 58,752 cached, value per byte 52,608, value per byte x
hit count also 52,608 and a loss on the incident shape). The c8 cell was not run on the final candidate: its
preview needed the 900 s deadline, and the c4 and c1 cells had already decided the order.

Not measured here: exactness of an agentic greedy run (the shipped change does not alter anything on this card;
the mixed shape's greedy rows cover the cached-versus-cold identity of every arm).
