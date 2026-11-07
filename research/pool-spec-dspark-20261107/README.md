# The spec and DSpark pools under the grid law (memra#1093 item 1, 2026-11-07)

Verdict: on the RTX 5090 Laptop at the device default the DSpark pool broke warm == cold, and the MTP spec pool did not. A three-turn DFlash chat with max_tokens 96 / 1024 / 512 resumed turn 3 from the parked turn-2 session at 3,114 of 3,144 tokens (rows the verify and decode wrote, off the prime grid) and left the cold text (383 vs 371 characters); so did 96 / 256 / 96 with the prefix cache off and MEMRA_AFFINITY=0. The fix frees a recurrent DFlash session at retire instead of parking it, and the probe refuses a recurrent hit as a backstop; both failing shapes are then byte-equal to cold. Chat and raw-completion replays at short budgets never reached either pool, and the MTP spec pool never resumed in any cell.

## Cell

`pool_pair_cells.py`: one warm boot serves a three-turn chat in order (each turn exactly extends the
previous prompt and answer), one cold boot per turn 2 and turn 3. Qwen3.8-27B NVFP4 + Q5_K GGUF,
context 8,192, two sessions, prefix cache at its derived size, switches at the device default (the
DFlash restore switch is off on the RTX 5090). Route `dspark` arms the DFlash2 drafter; `mtp` leaves it
off so the MTP spec route serves; `-raw` sends the turns as `/v1/completions` text (turn N+1 = turn N's
prompt + completion + a short follow-on). Per-turn max_tokens decide the route: DFlash prefers a cold
prime over a plain prefix hit once the decode budget is at least prompt / 8, and a turn-2 budget
that also covers turn 3's charged cap lets the pool's token tier take turn 3. Greedy, thinking off.
Character counts exclude the reasoning/content separator.

## Before (main's behavior: 05d93be269 = main 46172c1bac plus the rewind/decline counters)

| arm | route | max_tokens | env | turn 3 warm / cold | bytes | what the warm boot did on turns 2-3 |
|---|---|---|---|---|---|---|
| dspark-512 | dspark | 96 / 512 / 512 | default | 371 / 371 | pass | DSpark pool MISS (template seam), cold primes |
| mtp-96 | mtp | 96 / 96 / 96 | default | 371 / 371 | pass | prefix-cache spec restores at 2,976 and 3,072 |
| mtp-512 | mtp | 96 / 512 / 512 | default | 371 / 371 | pass | the same |
| dspark-raw-512 | dspark-raw | 96 / 512 / 512 | default | 157 / 157 | pass | DSpark pool MISS |
| mtp-raw-96, mtp-raw-512 | mtp-raw | 96 / 96 / 96, 96 / 512 / 512 | default | 157 / 157 | pass | prefix-cache spec restores at 2,944 and 3,040 |
| dspark-raw-512-nopc | dspark-raw | 96 / 512 / 512 | prefix cache off | 157 / 157 | pass | DSpark pool MISS |
| mtp-raw-96-nopc, mtp-96-nopc | mtp(-raw) | 96 / 96 / 96 | prefix cache off | pass | pass | no pool resume |
| dspark-512-nopc | dspark | 96 / 512 / 512 | prefix cache off | 371 / 371 | pass | DSpark pool MISS (turn 3's cap outgrew turn 2's) |
| **dspark-1024** | dspark | 96 / 1024 / 512 | default | **383 / 371** | **FAIL** | `dspark-reuse: 3114 committed tokens resumed` |
| mtp-1024 | mtp | 96 / 1024 / 512 | default | 371 / 371 | pass | prefix-cache spec restores |
| **dspark-256-nopc-aff0** | dspark | 96 / 256 / 96 | prefix cache off, MEMRA_AFFINITY=0 | **383 / 371** | **FAIL** | `dspark-reuse: 3114 committed tokens resumed` |
| dspark-raw-1024 | dspark-raw | 96 / 1024 / 512 | default | 157 / 157 | pass | DSpark pool MISS |
| mtp-1024-nopc-main | mtp | 96 / 1024 / 512 | prefix cache off | 371 / 371 | pass | no pool resume |

Receipts: `before-chat/`, `before-raw/`, `before-no-prefix-cache/`, `before-long-budget/` and
`after-c371/` (the MTP prefix-cache-off arm on main sits there too), each with per-turn JSON, the warm
and cold server logs and `verdicts.jsonl`.

Why the pool mostly misses: the parked stream holds the turn's answer as generated (and the template's
end-of-turn token), while a chat client re-renders the assistant turn, so the text tier diverges at the
seam (`dspark-reuse MISS: ... best text overlap 14368B of 14509B`). The token tier takes the turn when
the next prompt extends the parked tokens exactly and its cap fits the parked allocation: turn 2's
max_tokens 1024 (or 256 with the prefix cache off, where turn 2 runs DFlash cold) covers turn 3.

## Cause and change

A DFlash session parks with its trunk cache at the committed stream: rows past its own prompt were
written by speculative verify and decode, and its end is off the 32-token prime grid. Resuming it primes
the next suffix from state the cold prime never computes (the grid law, docs/SERVING.md). The plain pool
rewinds to a grid checkpoint (memra#1091); a DFlash session keeps none, and every DFlash target is a GDN
hybrid (`gdn_dspark_compatible`), so:

- the DSpark probe refuses a recurrent exact or text hit, drops the entry and serves cold
  (`dspark-reuse DECLINED (grid law)`, `/metrics` `dspark_pool_grid_declines`);
- a recurrent DFlash session is freed at retire instead of parked (`dspark-park SKIPPED (grid law)`,
  `dspark_pool_grid_park_skips`), so no pool slot or memory is held for a stream that cannot resume.

## After

On c371b5bf98 (the decline; the park skip came after review and is gated on the landing head):

| arm | turn 3 warm / cold | bytes | warm log |
|---|---|---|---|
| dspark-1024-fix | 371 / 371 | pass | `dspark-reuse DECLINED (grid law): the parked stream's 3114 rows ...` |
| dspark-256-nopc-aff0-fix | 371 / 371 | pass | the same |
| mtp-1024-nopc-fix | 371 / 371 | pass | no pool resume |

Turn 1's sha is identical in every arm and boot of a route.

## Price

`research/pool-resume-cost-20261107/run-3-dspark-long/` (the dspark-1024 shape, ABBA, 3 conversations
per boot): turn 3 TTFT 0.163 s on main (the inexact pool resume) against 2.745 s with the decline
(a cold DFlash prime of 3,144 tokens; median of 6, 2.60-2.82 s). The 5090 keeps
`MEMRA_DSPARK_PREFIX_RESTORE` off, so no on-grid entry with a draft tail exists to restore DFlash from;
with `MEMRA_DSPARK_PREFIX_RESTORE=1` the same turn restores DFlash from an on-grid entry carrying
the draft tail at 0.277 s (`run-4-switch/`), so re-pricing the 5090's switch default is the
memra#1093 follow-up. The path fired 0 times on the 5090-c8 lane's agentic workload
(`research/pool-resume-cost-20261107/agentic-c8/`).
