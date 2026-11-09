# Growing a rewound DFlash session to the next turn's cap (memra#1093 follow-up, 2026-11-08)

Verdict: positive and exact, ships (PR #1116). Growing a rewound DFlash session to the next turn's
cap and then seating it resumes every turn whose parked session survives admission, byte-equal to
cold in 15 of 15 pair cells (greedy, T=1 and penalized; chat, raw and chains; the new agentic and
cache-preferred shapes). On the RTX 5090's agentic workload at its device default it turns the 3 of
11 second turns at c1 that the cap refused into resumes (TTFT 2.6-3.0 s to 0.45-0.83 s; the cell's
summed TTFT -7%, end-to-end -5%) with the memory floor, refusals and output unchanged; c2 and above
are unchanged. The other 8 lost their session to the admission reclaim or the cache-preferred
veto: on top of the c8 round-2 lane's held prefill-workspace charge all 11 resume (second-turn
TTFT median 3.37 s to 1.41 s, end-to-end -13% at c1). #1103's turn 3 does not regress (0.263 to
0.258 s), and its turn 2, refused by the cap before, drops from 2.319 to 0.273 s.

PR #1103 parks a retiring recurrent DFlash session rewound to the last prime boundary `g` it
published itself, and the exact token tier resumes it as a new stream. On #1103's agentic memory
A/B the change parked 62 sessions rewound and resumed none (`../pool-rewind-dspark-20261108/mem/`).
This lane finds why and fixes it.

## Why the agentic turns did not resume (the mechanism)

Replayed on main d6ef68f5a1 (binary sha256 2e535649d379, the cell-evidence build of that commit) with
`grow_ab.sh`'s config, the 2026-11-05 decision cell's (Qwen3.8-27B NVFP4 + Q5_K, DFlash2 q4 drafter,
FR-Spec trim, `MEMRA_MTP_SKIP=1`, context 32,768, 8 sessions, prefix cache 4,000 MB, no host tier,
T=1 seed 1234, thinking on, 768 tokens), full workload (29 bodies: 11 agent sessions with two turns
and 7 single-turn long ones), one boot, c1 c2 c4 c8 (`confirm/`, 16:26-17:02). `agent_turns.py`
records each request and `trace_requests.py` joins it to its admission segment by response id.

At c1 each second turn follows its first turn directly, so its own session is the newest parked
entry. Of the 11 second turns:

| class | turns | what the log shows |
|---|---:|---|
| the admission reclaim evicted every parked DFlash session, the turn's own included, before the pool probe ran | 6 (k 1, 10, 17, 19, 21, 28) | `reclaim-on-defer: evicted ... 1-2 dspark parked sessions`, then no pool entry; the turn's cold cost (2.3-2.5 GB, 1.8-2.0 GB of it prefill workspace) needed the memory |
| the turn's own session was parked and its tokens prefixed the prompt, and the cap refused it | 3 (k 4, 7, 26) | `dspark-reuse MISS: 1 parked, best text overlap 7826B / 7933B / 8529B` (the session's `g` tokens); caps 3,769 / 3,829 / 3,464 against allocations of 3,099 / 3,235 / 3,199 |
| the cache-preferred veto served the turn plain from the boundary entry the session itself published | 2 (k 13, 24) | `[dspark] cache-preferred` (8,053 and 7,391-token prompts, decode budget 768 under the eighth), and the reclaim had evicted the session too |

47 rewound parks, 11 freed, 0 resumes (`dspark_pool_grid_park_rewinds` 47, `park_skips` 11,
`rewind_resumes` 0, `grid_declines` 0, `spec_pool_evictions` 45). At c2, c4 and c8 the second turns
go cold or to a prefix hit the same way, and often a second turn starts before its first turn has
parked (the replay runs bodies in order on N workers). A parked rewound session frees 804 MB
median when the reclaim evicts it (637-1,122 MB over 13 evictions with nothing else evicted); it
holds its trunk cache at its cap, the recurrent planes, the batched snapshot and the drafter's
ring.

The brief's cause, the cap, is the second class. The first class is memory: the reclaim's
headroom test charges the cold cost, and the parked session holds more than the per-session draft
charge it would replace. The prefill workspace charge is the large term (c8 round 2 found it
over-charged; see the combined arm below).

## The change

- **Grow first, then seat.** The DSpark pool's exact tier (`dspark_exact_pick`) takes a rewound
  session whatever the request's cap. After every cheaper refusal (sampler, the prime floor, the
  grid backstop) the resume arm reallocates it at the request's cold allocation
  (`HybridModel::dspark_grow_session`): a fresh `Cache::new` at the cap holding the full-attention
  rows `[0, g)` and the recurrent conv/ssm planes (`grow_trunk_cache`), the drafter's cap raised and
  its planes grown to the cold size when that is larger (`DflashKv::grow_cap`), the batched-snapshot
  tables dropped so the first round rebuilds them as on a cold session. Then `reseat_for_request`
  seats it. A CUDA OOM in the grow frees the unleased prefix entries and retries once; a grow or
  seat that fails serves cold and the entry goes.
- **The cache-preferred veto yields to the entry the probe would take**
  (`DsparkColdPrefixAdmission::pool_resumable`, through the same `dspark_exact_pick`): rewound,
  with the request's sampler, at least as deep as the plain hit or with a decode that repays the
  difference.
- **The plain pool decides its grid rewind before the park-compact grow**
  (`grid_rewind_plain_refusal`): a compacted recurrent entry the rewind would drop is dropped
  before the grow pays a request-sized allocation, a snapshot and the copies (memra#1093 item 4).
- Counter: `dspark_pool_grid_rewind_grows` (resumes that grew the session first, a subset of
  `dspark_pool_grid_rewind_resumes`).
- Tried and dropped: the admission reclaim evicting the arriving prompt's own rewound session
  after every other parked session. On an agentic replay of a build that had it (`diag/`, the
  change with that rule on the fix stack's cf8d1799f9, binary 72166a5b0814, one boot, 18:01-18:37)
  it kept the session 3 times, each a turn the global LRU kept on main too, and still had to evict
  it 18 times: it never changed which session survived, so it is not in the change.

## Exactness (`pairs/`)

Binaries: base = the landing base, the fix stack's tip 8633c4bf5d (sha256 973b74ed770c); fix = that
plus this change, 0191382652 (e7eab4d72aa6). `pool_pair_cells.py`: one warm boot serves a three-turn
chat in order, one cold boot primes turn 2 and one primes turn 3 from scratch; Qwen3.8-27B NVFP4 +
Q5_K GGUF + DFlash2, context 8,192, two sessions, prefix cache at its derived size, device defaults
(park compaction is the RTX 5090's default on this base), thinking off. A case passes only when
turns 2 and 3 are byte-equal to cold AND the warm log shows the expected path (`_EXPECT=rewind:P:R:G`:
at least P rewound parks, R prime-boundary resumes and G of them after a grow, every boundary on the
32-token grid, no grid-law or seat decline). Run 19:36-21:18 (`pairs/`, `pairs-base/`).

| case | route | max_tokens | sampling | fix: bytes, path | grows (alloc -> cap) | resumes at | base (fail-before) |
|---|---|---|---|---|---|---|---|
| dspark-1024-rw | chat | 96 / 1024 / 512 | greedy | pass, pass | 3,096 -> 4,141 | 2,976, 3,072 | path fails: turn 2 MISS on the cap, only turn 3 resumes |
| dspark-1024-t1-rw | chat | 96 / 1024 / 512 | T=1 | pass, pass | 1 | 2,976, 3,072 | |
| dspark-chain-rw | chat | 1536 / 448 / 448 | greedy | pass, pass | 0 (the allocation covers) | 2,976, 3,072 | |
| dspark-chain-t1-rw | chat | 1536 / 448 / 448 | T=1 | pass, pass | 0 | 2,976, 3,072 | |
| dspark-1024-t1-pen-rw | chat | 96 / 1024 / 512 | T=1, freq 0.3, pres 0.2 | pass, pass | 1 | 2,976, 3,072 | |
| dspark-chain-t1-pen-rw | chat | 1536 / 448 / 448 | T=1, penalties | pass, pass | 0 | 2,976, 3,104 | |
| dspark-raw-1024-rw | raw | 96 / 1024 / 512 | greedy | pass, pass | 1 | 2,944, 3,040 | |
| dspark-raw-chain-t1-rw | raw | 1536 / 448 / 448 | T=1 | pass, pass | 0 | 2,944, 2,976 | |
| dspark-256-nopc-aff0-rw | chat | 96 / 256 / 96 | prefix cache off, MEMRA_AFFINITY=0 | pass, pass (every park skipped) | - | - | |
| dspark-agentic-rw | chat | 512 / 512 / 512 | greedy | pass, pass | 2 | 2,976, 3,072 | path fails: 2 MISS, no resume |
| dspark-agentic-t1-rw | chat | 512 / 512 / 512 | T=1 | pass, pass | 2 | 2,976, 3,072 | path fails: 2 MISS, no resume |
| dspark-agentic-t1-pen-rw | chat | 512 / 512 / 512 | T=1, penalties | pass, pass | 2 | 2,976, 3,104 | |
| dspark-raw-agentic-t1-rw | raw | 512 / 512 / 512 | T=1 | pass, pass | 2 | 2,944, 2,976 | |
| dspark-veto-rw | chat | 256 / 256 / 256 | greedy | pass, pass | 2 | 2,976, 3,072 | path fails: the veto serves the session's own boundary entry plain |
| dspark-veto-t1-rw | chat | 256 / 256 / 256 | T=1 | pass, pass | 2 | 2,976, 3,072 | **bytes fail**: turn 3 warm 617 vs cold 651 characters (plain hit warm, DFlash cold) |

15 of 15 pass on the fix. The agentic shape (the same max_tokens every turn) grows on every resume.
The cache-preferred shape (a decode budget under the prompt's eighth) shows what the veto costs at
T=1 on the base: the warm turn takes the plain prefix hit while a cold boot primes DFlash, two
programs that draw different samples, so the warm text is not the cold one; with the change the
veto yields to the rewound session and the warm turn is the cold DFlash program.

## The agentic workload (`ab/`)

`grow_ab.sh` with the mechanism replay's config, base = the landing base 8633c4bf5d (973b74ed770c),
fix = this change (e7eab4d72aa6), full workload, one boot per arm running c1 c2 c4 c8, arms in the
order base, fix, fix, base, one gpulock lease per arm interleaved with another lane's leases
(2026-11-08 21:31 to 2026-11-09 00:42, `ab/`). Each cell lists its two runs.

| workload | c | arm | runs | tok/s | TTFT sum s | E2E sum s | 400 / 408 | 2nd-turn paths | 2nd-turn TTFT median s |
|---|---:|---|---:|---|---|---|---|---|---|
| full | 1 | base | 2 | 32.76 / 33.31 | 167.3 / 164.1 | 355.0 / 349.2 | 0 / 0 | cold 12, dspark-miss 6, plain-cache-preferred 4 | 3.518 / 3.516 |
| full | 1 | fix | 2 | 35.03 / 34.38 | 151.4 / 155.2 | 332.0 / 338.3 | 0 / 0 | cold 12, dspark-resume+grow 6, plain-cache-preferred 4 | 3.413 / 3.510 |
| full | 2 | base | 2 | 34.42 / 35.08 | 359.5 / 361.6 | 670.0 / 656.9 | 0 / 0 | cold 22 | 6.728 / 6.623 |
| full | 2 | fix | 2 | 35.42 / 34.99 | 355.5 / 341.6 | 650.9 / 659.3 | 0 / 0 | cold 22 | 6.516 / 7.646 |
| full | 4 | base | 2 | 33.15 / 32.06 | 888.7 / 920.8 | 1218.3 / 1259.2 | 0 / 0 | cold 19, prefix-hit 3 | 23.793 / 23.925 |
| full | 4 | fix | 2 | 32.52 / 32.80 | 896.5 / 877.1 | 1223.8 / 1216.5 | 0 / 0 | cold 19, prefix-hit 3 | 26.683 / 22.391 |
| full | 8 | base | 2 | 33.90 / 32.72 | 1446.7 / 1493.1 | 1744.9 / 1791.1 | 0 / 6 | cold 15, prefix-hit 7 | 66.933 / 65.378 |
| full | 8 | fix | 2 | 32.88 / 31.75 | 1381.8 / 1592.2 | 1664.6 / 1871.6 | 0 / 5 | cold 14, http-408 2, prefix-hit 6 | 52.780 / 65.828 |

- **Resumes:** 3 per fix run at c1, each a second turn whose cap outgrew its parked session (the
  cap class above, k 4, 7, 26), each grown first; 0 on the base. Their TTFT: 2.82 / 2.82, 3.00 /
  2.81 and 2.58 / 2.57 s on the base, 0.79 / 0.83, 0.72 / 0.73 and 0.46 / 0.45 s with the change.
  At c2 and above no second turn resumes in either arm: the reclaim evicted the session or the
  turn started before its first turn parked.
- **c1 totals:** TTFT summed over the 29 requests 167.3 / 164.1 s on the base, 151.4 / 155.2 s with
  the change; end-to-end 355.0 / 349.2 against 332.0 / 338.3 s; aggregate generation 32.76 / 33.31
  against 35.03 / 34.38 tok/s. c2, c4 and c8 overlap between the arms (both orders).
- **Memory:** the learned fixed residual reads 341 MB in every arm, no `out of memory` line, no
  HTTP 400 in any cell; c8 times out 3 + 3 requests (HTTP 408) on the base and 2 + 3 with the
  change, the same queue-deadline class #1103 reported.
- **Output:** every request that completed in both arms generated the same text (T=1, seeded):
  29 of 29 at c1, c2 and c4, 25 of 25 at c8.
- Counters per fix run: `dspark_pool_grid_rewind_resumes` 3, `dspark_pool_grid_rewind_grows` 3,
  `dspark_pool_grid_declines` 0; rewound parks 48-50 in every arm.

### On top of the held prefill-workspace charge (`combined/`)

The reclaim class is memory: the admission charges a cold prime's prefill workspace (1.8-2.0 GB of
a 2.3-2.5 GB turn cost on this workload), so the reclaim evicts the parked session the turn would
resume. A parallel lane (c8 round 2) changes that charge to the workspace a prime actually holds
(its commit 47e9d340a8, binary C1c, sha256 e0ee6417229a). Its base arm here is C1c; the fix arm is
this change replayed on that commit (25a763427a03). Same harness and config, arms in the order
base, fix, fix, base, one lease per arm (2026-11-09 01:04-04:18).

| workload | c | arm | runs | tok/s | TTFT sum s | E2E sum s | 400 / 408 | 2nd-turn paths | 2nd-turn TTFT median s |
|---|---:|---|---:|---|---|---|---|---|---|
| full | 1 | base | 2 | 35.20 / 36.12 | 153.4 / 147.8 | 330.4 / 322.0 | 0 / 0 | dspark-miss 18, plain-cache-preferred 4 | 3.372 / 3.142 |
| full | 1 | fix | 2 | 38.46 / 37.70 | 131.8 / 134.8 | 283.6 / 289.3 | 0 / 0 | dspark-resume+grow 22 | 1.411 / 1.425 |
| full | 2 | base | 2 | 37.69 / 38.45 | 185.7 / 183.7 | 620.3 / 608.7 | 0 / 0 | cold 6, dspark-miss 12, plain-cache-preferred 4 | 4.828 / 4.345 |
| full | 2 | fix | 2 | 37.50 / 37.68 | 175.9 / 183.6 | 586.1 / 583.6 | 0 / 0 | cold 6, dspark-miss 6, dspark-resume+grow 8, plain-cache-preferred 2 | 4.737 / 4.614 |
| full | 4 | base | 2 | 46.04 / 46.69 | 285.9 / 290.7 | 918.0 / 912.5 | 0 / 0 | cold 17, prefix-hit 5 | 6.110 / 5.542 |
| full | 4 | fix | 2 | 45.55 / 45.67 | 298.2 / 295.6 | 935.7 / 931.1 | 0 / 0 | cold 18, prefix-hit 4 | 5.641 / 5.666 |
| full | 8 | base | 2 | 47.56 / 45.46 | 1138.6 / 1081.0 | 1750.0 / 1718.3 | 0 / 0 | cold 20, prefix-hit 2 | 37.860 / 36.705 |
| full | 8 | fix | 2 | 44.08 / 44.93 | 1114.5 / 1110.4 | 1766.8 / 1747.1 | 0 / 0 | cold 20, prefix-hit 2 | 38.888 / 36.864 |

- **c1:** all 11 second turns resume, each grown first. On C1c alone 9 of them stay parked and
  miss on the cap, and 2 go plain under the cache-preferred veto, which now yields to the parked
  session. Second-turn TTFT median 3.37 / 3.14 s on C1c against 1.41 / 1.43 s; TTFT summed over
  the 29 requests 153.4 / 147.8 against 131.8 / 134.8 s; end-to-end 330.4 / 322.0 against 283.6 /
  289.3 s (-12%); generation 35.20 / 36.12 against 38.46 / 37.70 tok/s.
- **c2:** 4 resumes per fix run (the second turns whose first turn had parked); end-to-end 620.3 /
  608.7 against 586.1 / 583.6 s.
- **c4 and c8:** no resume or grow in any cell, and arms 2 to 4 take the same paths. c4 reads 46.04
  / 46.69 tok/s on C1c and 45.55 / 45.67 with the change (end-to-end +2%, both fix runs below both
  base runs); c8 overlaps. Since no code the change adds ran differently there, this reads as
  run-to-run variation under shifting host load (5-11 at the arms' starts); on the main base the c4
  cells overlap (33.15 / 32.06 against 32.52 / 32.80 tok/s).
- **Output:** identical texts for every request except the two vetoed second turns (k 13 and 24 at
  c1, k 13 at c2), which C1c serves plain from the prefix hit and the change serves by the DFlash
  resume, the cold program (the cache-preferred pair cells show the resume is byte-equal to a cold
  DFlash prime, and the plain hit is not at T=1). The two base arms agree with each other on every
  request.
- **Memory:** fixed residual 341 MB in every arm; no HTTP 400 or 408. One `out of memory` line, in
  the second fix arm's c2 cell: a step OOM on k 22 (a 16,643-token cold prime beside k 23's), with
  driver free at 36 MB after a prefix-cache boundary entry published between two prime calls of
  one tick; the step-OOM replay re-admitted k 22 and it completed. No grow ran in that tick (both
  turns missed the pool). That is the C1c defect the c8 round-2 lane reported and fixed after
  this binary.

## Turn-3 TTFT (`ttft/`)

`cost_cell.py` (from `../pool-resume-cost-20261107/`, plus the agentic shape), three distinct
conversations per boot, greedy, thinking off, streamed, base and fix interleaved ABBA, one boot per
arm, n = 6 per arm and turn (21:18-21:24, `ttft/`).

| shape | turn | base TTFT median (range) | fix TTFT median (range) | path on the fix |
|---|---:|---:|---:|---|
| #1103's long decode (96 / 1024 / 512) | 2 | 2.319 s (2.201-2.444) | 0.273 s (0.272-0.276) | turn 1's park grown 2,971 -> 4,003-4,014 rows, resumed at 2,848 |
| | 3 | 0.263 s (0.258-0.266) | 0.258 s (0.256-0.263) | resumed at 2,944 without a grow, as on the base |
| agentic (512 / 512 / 512) | 2 | 2.398 s (2.264-2.572) | 0.274 s (0.272-0.279) | grown 3,387 -> 3,491-3,502, resumed at 2,848 |
| | 3 | 2.535 s (2.324-2.667) | 0.268 s (0.262-0.274) | grown again (+35 rows), resumed at 2,944 |

Turn 1 is a cold prime in both arms (2.24 / 2.20 s and 2.34 / 2.44 s). Completion lengths are the
same in both arms. #1103's turn 3 does not regress; the agentic shape's turn-3 total time drops
from 3.81 s to 1.44 s.

## The 1..15-row affinity suffix

Left as it is, documented in docs/SERVING.md. Priming it as one batch is neither cheap nor exact: a
batched prime under `PRIME_MIN_T` rows needs a GEMM dispatch below m = 16 and a conv call below
`d_conv - 1` rows, and the cold prime folds such a tail into the range before it, so a short batched
call would still be another program. The exact alternative, the prime floor, primes the whole turn
cold. None of the plain-affinity rewinds in #1103's agentic receipts primed fewer than 16 rows.

## Scope

- One card (RTX 5090 Laptop), one model (Qwen3.8-27B NVFP4 + Q5_K with the DFlash2 drafter), the
  device default (the restore switch off, park compaction on). Switch-on devices (GB10, the Max-Q)
  park no rewound session, so the grow and the veto input never fire there.
- The grow serves exact token extensions of a rewound session; the text tier stays closed to
  rewound entries. A turn of fewer than `PRIME_MIN_T` new tokens still serves cold and keeps the
  entry.
- On the agentic workload at c2 and above the change rarely applies: the reclaim takes the parked
  sessions, and with N workers a second turn often starts before its first turn has parked. The
  replay is not causally ordered like a live agent, which only sends turn N+1 after turn N ends.
- Park compaction has no switch on this base, so the pair cells ran it on only; the DSpark path
  does not read it, and the one compaction-reachable change (the plain pool's decision before
  the grow) gives the same output by construction.
- The workload files and generated texts stay out of the repo (derived from private agent
  sessions): the request rows carry ids, lengths, timings and a text hash.

## Files

- `pool_pair_cells.py`, `pairs.sh`: the warm == cold pair cells (#1103's harness plus the grow's
  path checks and the agentic and cache-preferred shapes).
- `agent_turns.py`, `grow_ab.sh`, `trace_requests.py`, `summarize.py`: the agentic A/B with one row
  per request, joined to the server log's admission segment by response id.
- `cost_cell.py`: the TTFT cell harness of `../pool-resume-cost-20261107/` with the agentic shape
  added.
