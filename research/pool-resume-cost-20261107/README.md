# What the pool grid law costs where it sends a resume cold (memra#1093 items 2 and 5, 2026-11-07)

Verdict: on the RTX 5090 Laptop the pool grid law (memra#1091) costs about +0.1 s of TTFT on a 3k-token chat turn that the continuation pool used to resume at its parked end; the DSpark-pool decline (memra#1093) costs a long-decode DFlash turn its whole re-prime (TTFT 0.163 -> 2.745 s at 3k tokens), which MEMRA_DSPARK_PREFIX_RESTORE=1 wins back (0.277 s, by the DSPARK restore path #1083's switch-on pair cells found byte-equal on this card). Neither path fired on the 5090-c8 lane's agentic workload (where DFlash never served). On the default chat the turn now rewinds to the session's grid checkpoint (TTFT 0.114 -> 0.213 s); with MEMRA_AFFINITY=0, which arms no checkpoint, it is declined and served by the prefix cache's on-grid seed (0.110 -> 0.217 s). Raw completions never reached the plain pool in either arm (0.323 vs 0.322 s). The prefix cache's on-grid seeds carry the declined turns, so arming a checkpoint for every recurrent session would win back no more than that 0.1 s.

## Cell

`cost_cell.py`, base = main 05a8a63503 (before memra#1091), fix = 05d93be269 (main 46172c1bac plus the
pool rewind/decline counters), interleaved boots ABBA per shape. Qwen3.8-27B NVFP4 + Q5_K GGUF + DFlash2,
context 8,192, two sessions, switches at the device default. Each boot: one warm-up request, then 3
distinct three-turn conversations (distinct first line, no shared prefix), greedy, thinking off,
streamed, max_tokens 96 / 256 / 96 (turn 2's room lets its parked cap cover turn 3, so the continuation
pool takes turn 3, the memra#1091 shape). Per request the server log gives the pool resume, rewind or
decline and the prefix-cache hit (`runs.jsonl`).

## Turn 3 (6 requests per arm and shape)

| shape | base TTFT median (range) | fix TTFT median (range) | base path | fix path |
|---|---:|---:|---|---|
| chat, default | 0.114 s (0.104-0.138) | 0.213 s (0.210-0.216) | pool resume at the parked end (2,985 of 3,016) | rewind to the grid checkpoint, then resume |
| chat, MEMRA_AFFINITY=0 | 0.110 s (0.109-0.111) | 0.217 s (0.214-0.218) | pool resume at the parked end | declined (no checkpoint), prefix hit 2,944 |
| raw /v1/completions | 0.323 s (0.307-0.340) | 0.322 s (0.303-0.345) | prefix hit 2,912 (no pool resume) | the same |
| chat, MEMRA_AFFINITY=0, prefix cache off | 0.162 s (0.160-0.163) | 0.161 s (0.158-0.167) | DSpark-pool resume (`dspark-reuse: 2986 committed tokens resumed`) | the same |

Turn 2 is identical across arms (a prefix hit; 0.22 s; with the prefix cache off a cold DFlash prime,
2.46 s). Total time is not compared on turn 3: the base resumes on decode-written state and its greedy
answer differs from the cold one (memra#1091), so its completion lengths differ.

What the rows say:

- The cost is the re-prime from the grid point (2,944 here: the checkpoint rewind on the default chat,
  the prefix seed with MEMRA_AFFINITY=0) instead of from the parked end (2,975 to 2,986): 31 to 42 more
  rows (62 to 73 primed against 31), about 0.1 s.
- Raw completions re-tokenize the previous completion at the seam, so the plain pool (no text tier)
  never takes the turn; the grid law changes nothing there.
- With the prefix cache off, turn 2 runs DFlash cold and parks in the DSpark pool, which then resumes
  turn 3 at its parked end in both arms. That is item 1's exposure (`research/pool-spec-dspark-20261107/`).
- Not measured here: glm5_next trunks (no glm5_next model fits the 24 GB card) and DFlash sessions
  demoted to plain under load (needs a sustained load shape).

## Item 5, arming a checkpoint for every recurrent session

A checkpoint at the session's grid boundary would let the declined MEMRA_AFFINITY=0 turn rewind instead
of taking the prefix seed. Both re-prime from a grid point a few dozen rows before the prompt end, so the
win is bounded by the 0.1 s above. Arming costs a recurrent-state snapshot per session (about 157 MB on
this model) and takes raw-prompt sessions out of the in-batch fanout and prime-batch paths. Not armed.

## The DSpark-pool decline (memra#1093 item 2)

`run-3-dspark-long/`: the chat at the default with max_tokens 96 / 1024 / 512, so turns 2 and 3 run
DFlash and turn 3 is the DSpark pool's exact extension. Base = main's behavior (05d93be269), fix =
c371b5bf98 (the decline), ABBA, 3 conversations per boot.

| turn 3 | TTFT median (range), 6 requests | path |
|---|---:|---|
| base | 0.163 s (0.162-0.175) | `dspark-reuse: ... committed tokens resumed` at the parked end (inexact) |
| fix | 2.745 s (2.599-2.819) | `dspark-reuse DECLINED (grid law)`, cold DFlash prime of the whole prompt |

`run-4-switch/`: the same shape on the fix binary with the restore switch at the 5090 default (OFF) and
with `MEMRA_DSPARK_PREFIX_RESTORE=1`, ABBA:

| shape | turn 2 TTFT | turn 3 TTFT | path |
|---|---:|---:|---|
| switch default (OFF) | 2.608 s | 2.688 s | cold DFlash primes |
| switch ON | 0.282 s | 0.277 s | DSPARK restore from on-grid entries (2,848, then 2,944) |

With the switch on, the DFlash turn restores from an on-grid prefix entry that carries the draft tail
and primes the rest (the path #1083's switch-on pair cells found byte-equal on this card; this cell
records timing, not text); it recovers the speed the decline gave up, and turn 2 too. The 5090
default is OFF for memory (memra#1069); re-pricing that default is the memra#1093 follow-up.

## How often the paths fire on the agentic workload

`agentic-c8/`: the 5090-c8 lane's agentic cell (18 sessions, 8 users, 29 turns, T=1, 768 tokens,
context 32,768) on the fix binary c371b5bf98, once at its own configuration (MTP) and once with DFlash
armed. `counts.tsv` holds the server-log counts: 0 pool resumes, 0 plain rewinds or refusals, 0 DSpark
declines in both runs; the prefix cache served 34 and 44 hits. Under that memory pressure DFlash never
served a request (0 `dspark-acc` lines; 8 of 29 measured requests ended 408), so that arm says nothing
about DFlash traffic, and the agentic client re-sends each body with its own history, so neither pool
sees an exact extension.
