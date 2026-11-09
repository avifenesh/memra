# c8-workspace-20261108 results

Lane README: [`README.md`](README.md). Decision records:
[`docs/decisions/PREFILL-WORKSPACE-HELD.md`](../../docs/decisions/PREFILL-WORKSPACE-HELD.md) and the
2026-11-08 update in [`docs/decisions/BATCH-OOM-RECOVER-DEFAULT.md`](../../docs/decisions/BATCH-OOM-RECOVER-DEFAULT.md).
All cells ran on one RTX 5090 Laptop GPU (24 GB) in the balanced platform profile, with a fresh
boot per run and one gpulock lease per group of consecutive arms. Other lanes' cells ran between
leases, so the window is shared, not quiet. The scored phases (sections 3 to 7) compare
A = main `0c25263389` against B = this lane rebased on it (`a99a7a13ef`, the tree of the landed
commits less the later rebase onto main). Sections 8 and 9 hold the versions before it.

| section | what | verdict |
|---|---|---|
| 1 | what a prime holds, against what admission charged | the charge was 66-80% workspace; a call holds 290 KB per row beside the slab |
| 2 | item 2: the batched-chunk OOM retry on the 5090 Laptop | every `on` boot 9 of 9, every `off` boot lost its chunk |
| 3 | agentic c8 | measured-pass first token p50 22.0 -> 14.5 s, p99 64.7 -> 39.1 s; prime-pass p50 31.0 -> 39.3 s |
| 4 | greedy c8 | 29 + 29 byte-identical |
| 5 | the mixed shape | flat, 103 of 103 byte-identical |
| 6 | c4 and c1 | flat |
| 7 | the DFlash route (the grow-rewind replay) | no OOM; c1 11 of 11 second turns resume (3 on main); c8 no 408 (5 on main) |
| 8 | the prime-pass tail and the deadline estimate | the slowest first token falls from 86-89 s to 60-63 s |
| 9 | the versions before the final tree | history |
| 10 | item 4: the census on git 2.43 | the GB10 answers as the rig does |

## 1. What a prime holds (`runs/probe/`, the measurement build of main `2288c4bf82`)

What admission charged a plain request on main (the `[admission] request cost` lines of
`runs/probe/c8/serve.log`; `MB` = 1e6 bytes):

| request ctx (prompt + 768) | prefill workspace charged | of the request's charge |
|---|---|---|
| 1,669 | 769 MB | 66% |
| 4,808 | 1,905 MB | 80% |
| 8,223 | 2,030 MB | 78% |
| 13,297 | 2,150 MB | 75% |
| 17,933 | 2,229 MB | 72% |

What the prime calls held (`*/ws-probe.json`, `*/ws-probe.md`; peak = pool `USED_MEM_HIGH` during
the call less `USED_MEM_CURRENT` at its start):

| prime call | calls | peak growth during the call | per row |
|---|---|---|---|
| agentic c8, plain 1024-row tick, slab resident (4096 rows) | 306 | median 292 MB, max 297 MB | 290 KB |
| agentic c1 prime pass, 4096-row MTP walker chunk, slab resident | 24 | median 1,170 MB, max 1,187 MB | 290 KB |
| mixed shape, concat batch of 2-6 fresh prompts (at most 512 rows) | 42 | median 67 MB, max 171 MB | 444 KB |
| boot calibration, 4096-row MTP chunk, slab 2048 -> 4096 rows | 1 per boot | 1,942 MB, 763 MB of it the new slab | 288 KB beyond the growth |

- The slab grew in exactly two calls per boot, both before serving (the wave probe's first prime
  to 2048 rows, then the calibration's MTP chunk to 4096 rows: 1,527 MB on main's seven-plane slab).
  It never grew while serving.
- A plain call's retained growth was 21 MB per 1024 rows, its own returned hiddens
  (1024 x 5,120 x 4 bytes), freed by its caller. No plain call held a prompt-long stack.
- No OOM line in the three boots; lowest device free 1,619 MiB (c8), 2,003 MiB (c1), 1,267 MiB
  (mixed).

## 2. Item 2: the batched-chunk OOM retry on the 5090 Laptop (`runs/oom-main/`)

DAY49 addendum D's aimed serving shape (`research/spill-b-20260919/day49d-run.sh`), all eight boots
on main `2288c4bf82` through the last tree with the env door: the 9B, `MEMRA_CTX=65536`,
`MEMRA_STEP_OOM_FAULT=batch:1`, burst 8 x 6,144 tokens, no warm, N=5; P boots with
`MEMRA_SERVE_SPEC=0`. `read-serve.log`, verbatim:

```
DAY49D SERVE card=rtx5090 boot=O1-off arm=off route=spec-default batch_fired=[8] other_fired=0 retry_lines=0 retried_ok=0 retry_failed=0 rows=9 status={200: 1, 503: 8} error_rows=8 -> READING (the before)
DAY49D SERVE card=rtx5090 boot=O1-on arm=on route=spec-default batch_fired=[2] other_fired=0 retry_lines=1 retried_ok=1 retry_failed=0 rows=9 status={200: 9} error_rows=0 -> PASS
DAY49D SERVE card=rtx5090 boot=O2-off arm=off route=spec-default batch_fired=[8] other_fired=0 retry_lines=0 retried_ok=0 retry_failed=0 rows=9 status={200: 1, 503: 8} error_rows=8 -> READING (the before)
DAY49D SERVE card=rtx5090 boot=O2-on arm=on route=spec-default batch_fired=[2] other_fired=0 retry_lines=1 retried_ok=1 retry_failed=0 rows=9 status={200: 9} error_rows=0 -> PASS
DAY49D SERVE card=rtx5090 boot=P1-off arm=off-plain route=plain batch_fired=[8] other_fired=0 retry_lines=0 retried_ok=0 retry_failed=0 rows=9 status={200: 1, 503: 8} error_rows=8 -> READING (the before)
DAY49D SERVE card=rtx5090 boot=P1-on arm=on-plain route=plain batch_fired=[8] other_fired=0 retry_lines=1 retried_ok=1 retry_failed=0 rows=9 status={200: 9} error_rows=0 -> PASS
DAY49D SERVE card=rtx5090 boot=P2-off arm=off-plain route=plain batch_fired=[2] other_fired=0 retry_lines=0 retried_ok=0 retry_failed=0 rows=9 status={200: 7, 503: 2} error_rows=2 -> READING (the before)
DAY49D SERVE card=rtx5090 boot=P2-on arm=on-plain route=plain batch_fired=[8] other_fired=0 retry_lines=1 retried_ok=1 retry_failed=0 rows=9 status={200: 9} error_rows=0 -> PASS
DAY49D SERVE-ROUTE card=rtx5090 route=plain boots=4 reached=4
DAY49D SERVE-ROUTE card=rtx5090 route=spec-default boots=4 reached=4
```

Every boot reached a batched chunk of at least two sessions. Every `on` boot answered 9 of 9
with one retry and `retried (ok)`; every `off` boot lost its faulted chunk's sessions. This is the
target card's reading (DAY49 2.4). The O1-off boot's lease ran 21 minutes, not 3, because the
harness's server stop looks for a process named `memra-server` and the first binary was staged
under another name; the boot's rows were complete when it was stopped by hand, and the later boots
ran from a binary named `memra-server`.

## 3. Item 1: agentic c8 (`runs/c8/`, A = main `0c25263389`, B = the lane on it, A B B A A B)

Agentic c8 is the c8 lane's shape: 29 turns of 18 coding-agent sessions, prompts of 1.4k-17k tokens, thinking on, T=1, 768 new tokens. Each run has two passes:

- **Prime pass:** `max_tokens=1`, all 29 turns at once, every prompt cold.
- **Measured pass:** the full turns.

Every run is a fresh boot, and the six runs ran in one lease. Percentiles pool each arm's three runs (87 requests per pass); tok/s and requests/s are the median run. TPOT = (E2E - TTFT) / (tokens - 1) per request, since this harness records no ITL.

| agentic c8, three runs per arm | A (main) | B (held charge) |
|---|---|---|
| failed requests, both passes | 0 of 174 | 0 of 174 |
| measured pass: first token p50 / p95 / p99 | 22.0 / 52.7 / 64.7 s | 14.5 / 34.8 / 39.1 s |
| measured pass: E2E p50 / p95 / p99 | 53.7 / 109.7 / 133.2 s | 52.7 / 122.4 / 149.2 s |
| measured pass: TPOT p50 / p95 / p99 | 96 / 167 / 181 ms | 135 / 224 / 322 ms |
| measured pass: tok/s, requests/s (median run) | 47.76, 0.130 | 49.52, 0.135 |
| prime pass: first token p50 / p95 / p99 | 31.0 / 67.1 / 75.4 s | 39.3 / 55.4 / 62.6 s |
| prime pass: slowest first token per run | 73.8, 75.4, 66.0 s | 60.0, 62.1, 62.6 s |
| prime-pass turns past 85 s / past 90 s | 0 / 0 | 0 / 0 |
| VRAM defers / reserve yields / prime-call yields | 4,303 / 7 / 0 | 1,922 / 52 / 0 |
| prefix hits (cached tokens), both passes | 86 (55,200) | 47 (64,416) |
| deadline-order ticks / OOM lines / batch OOM retries | 9 / 0 / 0 | 16 / 0 / 0 |
| lowest driver-free sample (250 ms) | 947 MiB | 307 MiB |

**Measured pass.**

- The first token comes sooner: 34% at the median and 40% at p99.
- Aggregate throughput rises 3.7%. Whole requests end at the same median: E2E p50 is 52.7 s against 53.7 s.
- Per-token decode is slower (TPOT p50 +41%), because more sessions decode together.
- VRAM defers more than halve.

**Prime pass.** The slowest first token falls in every run: 60-63 s against 66-75 s. The cost is the burst's median, 39.3 s against 31.0 s (+27%). The deadline order engages earlier now (section 8). In a burst whose deadlines are nearly equal, its earliest-deadline order is close to oldest first, so short prompts behind long ones wait longer.

**The free floor** is driver free between a seed insert and the next yield. No allocation failed in any run.

## 4. Greedy c8: exactness (`runs/g8/`, A = main, B = the lane)

Agentic c8 at T=0 with 256 new tokens, one run per arm (`exact_cmp.py`, the sha256 of each turn's output). Between A and B, all 29 prime-pass outputs and all 29 measured-pass outputs are byte-identical. No run logged an OOM line.

## 5. The mixed shape (`runs/mixed/`, A = main, B = the lane, A B B A A B)

This is the `spec-cache-mixed` shape with `MEMRA_MAX_SESSIONS=12`: greedy, short and long requests, and repeated prefixes (`score.py`).

| run | tok/s | TTFT p50 ms | rows | headroom evictions | lowest free MiB |
|---|---|---|---|---|---|
| 01-A | 125.90 | 924 | 96 | 1 | 1,331 |
| 02-B | 124.58 | 921 | 96 | 1 | 1,395 |
| 03-B | 124.83 | 932 | 96 | 1 | 1,395 |
| 04-A | 117.99 | 953 | 91 | 2 | 1,363 |
| 05-A | 120.64 | 976 | 96 | 1 | 1,363 |
| 06-B | 121.52 | 991 | 96 | 1 | 1,395 |

- **Throughput:** flat. The median run is 124.58 tok/s for B and 120.64 for A, and the ranges overlap (A 117.99-125.90, B 121.52-124.83).
- **First token:** TTFT p50 is 932 ms for B and 953 ms for A.
- **Evictions and OOM:** headroom evictions are 3 for B and 4 for A. No run logged an OOM line.
- **Exactness:** each B run matches A's first run on all 103 rows, byte for byte.

## 6. Lower load: agentic c4 and c1 (`runs/c4/`, `runs/c1/`, A = main, B = the lane)

This is the agentic c8 workload with fewer users. The c4 phase runs A B B A A B, and the c1 phase runs A B.

| agentic c4, three runs per arm | A (main) | B (held charge) |
|---|---|---|
| failed requests, both passes | 0 of 174 | 0 of 174 |
| measured pass: first token p50 / p95 / p99 | 3.1 / 15.2 / 19.9 s | 3.1 / 15.1 / 19.8 s |
| measured pass: E2E p50 / p95 / p99 | 27.2 / 59.0 / 67.9 s | 27.2 / 58.9 / 66.4 s |
| measured pass: TPOT p50 / p95 / p99 | 73 / 112 / 138 ms | 72 / 111 / 140 ms |
| measured pass: tok/s per run | 47.55, 50.21, 49.85 | 48.12, 49.34, 50.08 |
| prime pass: first token p50 / p99 | 11.7 / 48.6 s | 11.9 / 48.4 s |
| prime pass: slowest first token per run | 47.8, 48.6, 47.2 s | 47.4, 48.4, 46.7 s |
| VRAM defers / reserve yields / prime-call yields | 0 / 0 / 0 | 0 / 11 / 0 |
| deadline-order ticks / OOM lines | 0 / 0 | 0 / 0 |
| lowest driver-free sample | 1,619 MiB | 723 MiB |

At c4 the cold charge never deferred a request, so the held charge has nothing to admit sooner. The two arms are the same within run-to-run spread. The deadline order never engaged in either arm.

| agentic c1, one run per arm | A (main) | B (held charge) |
|---|---|---|
| failed requests, both passes | 0 of 58 | 0 of 58 |
| measured pass: first token p50 / p95 / p99 | 2.4 / 14.0 / 14.7 s | 2.3 / 13.7 / 14.1 s |
| measured pass: E2E p50 / p95 / p99 | 6.9 / 17.8 / 20.8 s | 6.8 / 17.2 / 20.4 s |
| measured pass: TPOT p50 | 14.5 ms | 14.2 ms |
| measured pass: tok/s | 33.84 | 34.53 |
| prefix hits (cached tokens), both passes | 52 (71,872) | 54 (72,384) |
| outputs, both passes | 58 | 58 byte-identical to A (same seed) |
| OOM lines / lowest driver-free sample | 0 / 1,555 MiB | 0 / 1,171 MiB |

At c1 one session holds the card, and both arms take the same paths. The held charge changes nothing a single session can see.

## 7. The DFlash route: the grow-rewind lane's agentic replay (`runs/grow/`, A = main, B = the lane)

This is `grow_ab.sh` from `research/pool-grow-dspark-20261108/`. It runs Qwen3.8-27B with the DFlash2 drafter and FR-Spec trim (`MEMRA_MTP_SKIP=1`), at 32,768 context, 8 sessions and a 4,000 MB prefix cache, T=1, thinking on, 768 tokens. The full agentic workload runs at c1, c2, c4 and c8 in each boot. The arms ran base, fix, fix, base; the first two in one lease, and the last two after another lane's landing record took the 5090 between them.

Main (`0c25263389`) already has grow-rewind (#1116). This phase checks the DFlash walkers under the held charge, the route the c8 shapes above do not take. The grow lane's combined arm on `C1c` logged one step OOM at c2 here.

| arm | c | answered | 400 / 408 | gen tok/s | TTFT sum s | E2E sum s | 2nd-turn paths | 2nd-turn TTFT median s |
|---|---:|---|---|---|---|---|---|---|
| 1-base-full | 1 | 29 of 29 | 0 / 0 | 36.38 | 144.9 | 319.7 | cold 6, dspark-resume+grow 3, plain-cache-preferred 2 | 3.288 |
| 2-fix-full | 1 | 29 of 29 | 0 / 0 | 38.1 | 133.46 | 286.26 | dspark-resume+grow 11 | 1.34 |
| 3-fix-full | 1 | 29 of 29 | 0 / 0 | 37.07 | 138.07 | 294.23 | dspark-resume+grow 11 | 1.451 |
| 4-base-full | 1 | 29 of 29 | 0 / 0 | 35.53 | 150.0 | 327.4 | cold 6, dspark-resume+grow 3, plain-cache-preferred 2 | 3.396 |
| 1-base-full | 2 | 29 of 29 | 0 / 0 | 35.65 | 356.98 | 647.3 | cold 11 | 6.853 |
| 2-fix-full | 2 | 29 of 29 | 0 / 0 | 37.64 | 175.22 | 583.67 | cold 3, dspark-miss 2, dspark-resume+grow 5, plain-cache-preferred 1 | 3.964 |
| 3-fix-full | 2 | 29 of 29 | 0 / 0 | 37.57 | 172.65 | 599.9 | cold 3, dspark-miss 1, dspark-resume+grow 6, plain-cache-preferred 1 | 3.521 |
| 4-base-full | 2 | 29 of 29 | 0 / 0 | 35.67 | 359.59 | 646.74 | cold 10, plain-cache-preferred 1 | 6.718 |
| 1-base-full | 4 | 29 of 29 | 0 / 0 | 35.52 | 806.58 | 1123.22 | cold 10, prefix-hit 1 | 21.194 |
| 2-fix-full | 4 | 29 of 29 | 0 / 0 | 45.71 | 298.63 | 933.5 | cold 9, prefix-hit 2 | 5.601 |
| 3-fix-full | 4 | 29 of 29 | 0 / 0 | 45.44 | 295.58 | 934.29 | cold 9, prefix-hit 2 | 5.649 |
| 4-base-full | 4 | 29 of 29 | 0 / 0 | 33.85 | 844.34 | 1177.15 | cold 9, prefix-hit 2 | 21.61 |
| 1-base-full | 8 | 26 of 29 | 0 / 3 | 33.21 | 1347.44 | 1631.87 | cold 7, http-408 1, prefix-hit 3 | 54.854 |
| 2-fix-full | 8 | 29 of 29 | 0 / 0 | 45.12 | 1112.23 | 1748.14 | cold 10, prefix-hit 1 | 36.657 |
| 3-fix-full | 8 | 29 of 29 | 0 / 0 | 45.94 | 1146.54 | 1756.39 | cold 8, plain-pool 1, prefix-hit 2 | 43.249 |
| 4-base-full | 8 | 27 of 29 | 0 / 2 | 37.06 | 1436.6 | 1728.1 | cold 8, prefix-hit 3 | 61.868 |

| arm | lowest driver free, 250 ms telemetry | lowest driver free the server logged (before a trim or settle) | lowest after an admission trim | OOM lines | step-OOM parks | rewound resumes / grows |
|---|---|---|---|---|---|---|
| 1-base | 871 MiB | 484 MiB | 2,020 MiB | 0 | 0 | 3 / 3 |
| 2-fix | 519 MiB | 132 MiB | 1,796 MiB | 0 | 0 | 18 / 18 |
| 3-fix | 487 MiB | 100 MiB | 1,508 MiB | 0 | 0 | 18 / 18 |
| 4-base | 839 MiB | 452 MiB | 2,020 MiB | 0 | 0 | 3 / 3 |

- **c1:** on B, all 11 second turns resume a grown session, against 3 on A. Their first token median is 1.34 / 1.45 s against 3.29 / 3.40 s, and generation is 38.10 / 37.07 tok/s against 36.38 / 35.53.
- **c2:** TTFT summed over the 29 requests is 175 / 173 s against 357 / 360 s.
- **c4:** generation is 45.71 / 45.44 tok/s against 35.52 / 33.85, and TTFT summed is 299 / 296 s against 807 / 844 s. The cold charge deferred these turns (VRAM defer lines over a boot's four cells: 13,394 in the first base arm, 4,253 in the first fix arm), and the held charge admits them.
- **c8:** B answers 29 of 29 in both runs. A answers 26 and 27: five HTTP 408s, its first-token deadline. Generation is 45.12 / 45.94 tok/s against 33.21 / 37.06.
- **Memory:** no run has an OOM line or a step-OOM park. The lowest readings the server logged are the before side of an admission trim (`[admit-trim] ... driver_free_bytes=104988672 ... driver_free_after=2118254592 before prime`) and of a prefix-reserve settle. In both cases the pool held cached blocks, and the trim or settle returned them before the prime or step ran. After a trim the lowest reading was 1,508 MiB.

## 8. The prime-pass tail and the deadline order's estimate (`runs/prime-c1e-c1g/`, `runs/c8-c1g/`)

On `C1e`, the held charge without the estimate change, the cold-burst prime pass put its slowest first token at 86-89 s in every run. Every one was the 17,101-token prompt (`runs/c8-c1e/`, section 9). The cause is not admission. That request was admitted 53-73 ms after it was sent, as the sixth to eighth session (`active=6 wave=8`), and then shared prefill eight ways.

The deadline-aware prefill order (`deadline_prefill_rows`, `docs/decisions/C8-SERVING-24GB.md`) switches to earliest deadline first when a row would miss its deadline under the round robin. It used the replacement estimate only while requests waited in the admission queue. The held charge admits the burst, so the queue was empty, and the order fell back to the form that lets finishing peers leave. The c8 record had measured that form 2x short (31 s predicted, 69 s taken, one run). In C1e's prime-only runs the order never switched.

The estimate now keeps the tick's current takes for every take the row still needs (`C1g`, then the final tree). The prime pass alone, A = `C1e`, B = `C1g`, A B B A A B (`prime_tail.py`):

| run | failed | first token p50 s | slowest s (prompt tokens) | over 85 s | over 90 s | deadline-order switches (whole run) | OOM lines |
|---|---|---|---|---|---|---|---|
| 01-A | 0 of 29 | 22.5 | 86.1 (17,101) | 1 | 0 | 0 | 0 |
| 02-B | 0 of 29 | 36.4 | 59.8 (16,643) | 0 | 0 | 3 | 0 |
| 03-B | 0 of 29 | 38.9 | 62.5 (16,643) | 0 | 0 | 3 | 0 |
| 04-A | 0 of 29 | 23.4 | 86.5 (17,101) | 1 | 0 | 0 | 0 |
| 05-A | 0 of 29 | 24.8 | 88.1 (17,101) | 1 | 0 | 0 | 0 |
| 06-B | 0 of 29 | 39.6 | 61.1 (16,643) | 0 | 0 | 3 | 0 |

- **The tail:** the slowest first token falls from 86-88 s to 60-63 s.
- **The cost, the burst's median:** 22.5-24.8 s rises to 36.4-39.6 s. In a burst whose deadlines are nearly equal, earliest deadline first is close to oldest first, so a short prompt behind a long one waits longer.
- **The measured pass is not hurt:** first token p50 / p99 is 14.1 / 36.9 s on `C1g` (`runs/c8-c1g/`, two runs per arm against `S`), against 15.1 / 36.7 s on `C1e`.

The independent review's simulation of decode-heavy turns predicted 21.6-24.2 s for that median. The measurement does not show it.

## 9. The versions before the final tree

Each version below ran as B against `S` (the fix stack's tip) on the same harness. None of them is scored. They are the history behind the record's Rejected list.

### `C1b` (`b868464d9b`): the reserve yield only before the decode step (`runs/c8-v2/`)

Three pairs of agentic c8. The measured pass's first token p50 was 14.6 s, against 23.4 s for A.

One B run (`06-B`) failed four requests with `class=Overloaded prefill error: DriverError(CUDA_ERROR_OUT_OF_MEMORY)`: three in the prime pass and one in the measured pass. Each time, seeds of 1,538 and 1,507 MB had published within a few seconds of one prefill phase. They took driver free to 52 MB, and the next prime call failed.

### `C1c` (`47e9d340a8`): the whole reserve yielded before every prime call and batch round (`runs/c8-c1c/`, `g8-c1c/`, `mixed-c1c/`)

- **Agentic c8:** no failures and no OOM line. Measured-pass first token p50 / p99 was 16.0 / 39.0 s against 22.0 / 54.7 s. The prime pass's slowest first token was 84.7-88.1 s.
- **Greedy pair:** 29 + 29 outputs byte-identical.
- **Mixed shape:** median 114.13 against 118.27 tok/s, with overlapping ranges (A 114.85-120.19, B 111.69-117.45). On the final tree the mixed shape is flat (section 5).

### `C1d` (`bb7f308cef`): the per-call yield at the batch formula (`runs/c8-c1d/`)

The yield before a plain prime call gave back only 458 KB per row of its take. Phase (a) yielded the reserve only while an MTP session was priming.

Two arms ran (`01-A`, `02-B`) before an independent review found three gaps:

- DFlash walkers prime in phase (a) with no `spec` state, so they had lost the yield;
- the fanout leader prime is a plain slab prime and had no yield;
- the need at the 8192-row solo widening was 3.76 GB, about three times what that call holds.

### `C1e` (`363e888249`): the yields as landed, the old deadline estimate (`runs/c8-c1e/`, `g8-c1e/`, `mixed-c1e/`)

- **Agentic c8:** no failures and no OOM line. Measured-pass first token p50 / p99 was 15.1 / 36.7 s against 21.0 / 51.2 s. tok/s was 50.41 against 48.23.
- **The prime pass's slowest first token:** 87.5, 89.3 and 86.0 s (section 8).
- **Greedy pair:** 29 + 29 outputs byte-identical.
- **Mixed shape:** 122.16 against 122.19 tok/s.

## 10. Item 4: the include census on git 2.43 (#1109)

The census reads the files the crates include at a commit with `git cat-file --batch`. It passed `--no-lazy-fetch`, which git 2.43 does not know, and the GB10 units run git 2.43. There it exited 129, and `cell_evidence.py` fell back to a whole checkout.

Before 2.45, the reader now runs the batch with `GIT_ALLOW_PROTOCOL=none` instead. A lazy fetch then has no transport. A blob the clone does not hold reads `missing`, and the reader refuses it. A partial clone whose needed objects are all local still answers.

Checked on a GB10 unit (git 2.43.0) at main `2288c4bf82`, against the rig (git 2.53.0):

| | rig, git 2.53, either tools | GB10, git 2.43, main's tools | GB10, git 2.43, #1109 |
|---|---|---|---|
| include census | 22,952 kept, sha256 `45a27644..` | none: `cat-file --batch` exit 129, whole checkout | 22,952 kept, sha256 `45a27644..` |
| sparse patterns | 35, sha256 `4ac2d405..` | (whole tree) | 35, sha256 `4ac2d405..` |
| inputs digest | `ff42b2ef..` | `ff42b2ef..` | `ff42b2ef..` |
| research_drift 30aae1b8c4..2288c4bf82 | 117 paths, sha256 `fd2187df..` | none | 117 paths, sha256 `fd2187df..` |
| cell tree | 824 MB sparse | whole tree | 824 MB sparse |

`cell_evidence.py need` on the GB10 names the same cells as on the rig. The comparison and its scripts are in the #1109 comment and `harness/census_check.py` and `harness/spark_census*.sh`.

## 11. Item 3: the two test fixes (no device run)

- `background_with_stream_is_a_contradiction_only_once_the_door_is_open` proves what its name says. With the door open, the refusal is the contradiction. With it closed, the refusal is background's own, never the contradiction.
- The `MEMRA_BACKGROUND_RESPONSES` gate command in docs/FLAGS.md (`cargo test -p memra-server --lib -- background translate_refuses_stateful`) now selects every test that takes the door's env lock. Six were renamed `background_*`, and a source scan fails when a lock taker falls outside the filter.
- `tools/test_local_build_cache.py` and `tools/test_release_input_view.py` hide Cargo configs above a TMPDIR fixture, as #1101 did for `test_release_inputs.py`. Under a home TMPDIR, 57 errors became passes.
