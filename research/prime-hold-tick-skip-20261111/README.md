# The batch-formation hold waits inside the tick instead of skipping it (memra#1223)

Date: 2026-11-11. Found while measuring memra#1211.

## Conditions

- GB10 (one DGX Spark unit per boot), Qwen3.8-27B NVFP4 + Q5_K with its MTP head, context 8,192, eight sessions, `MEMRA_TICK_TRACE=1` in every boot. Pass/fail style, single runs per boot, one boot per lease.
- Binaries, both built on the same unit:
  - `main2`: main ede1b8119f, `memra-0.141.1-0c5eb68559cb`.
  - `grace2`: the fix 7c8028cd17, `memra-0.141.1-b87b16188411`.
- Cells (`tools/stall_cell.py`): a long greedy streamed request runs alone for 8 s. Then closed-loop clients send one-token chat requests for 30 s, each waiting the think time after every answer. Then the long request runs alone for 12 s. The rows carry the long stream's inter-token gaps (p50 and p99 inside the window). `tools/normal_cell.py` with `STREAM=1`: c = 1, 2, 4, 8 clients, three greedy 160-token streamed requests each, with TTFT and inter-token gaps.
- `tools/primeticks.py` counts, for every plain admission, the ticks from its admission to the tick that primes it. 1 means its admission tick. The waits (`form_waits`, `form_wait_ms`) come from the tick trace; `tools/abtables.py` builds the tables below from the rows and the logs.

## The diagnosis (`gb10/diag/`)

memra#1211's stall-mark form (dd243cf79c on 83d37f5179) admits one-token requests to plain decode beside a demoted long stream. At 2 clients and 200 ms think, its short p50 was 0.476 s against main's 0.318 s.

| boot | 2:200 long tokens / short p50 / p95 | plain primes, ticks to prime | 2:0 short p50 |
|---|---|---|---|
| #1211 form, default holds | 262 / 0.476 / 0.479 | 3rd tick for 85 of 86 | 0.301 |
| the same, `MEMRA_PRIME_BATCH_HOLD_MS=0` | 224 / 0.327 / 0.329 | 1st for 110 of 110 | 0.301 |
| main, `MEMRA_SPEC_K=0` | 270 / 0.246 / 0.249 | 1st for 136 of 136 | 0.299 |
| main, default (shorts speculative) | 142 / 0.318 / 0.324 | none plain | 0.458 |
| the #1201 slot-arm probe (2628ece940) | 267 / 0.248 / 0.252 | 1st for 134 of 134 | 0.301 |

- Two windows of `MEMRA_PRIME_BATCH_HOLD_MS` are behind the 3rd-tick primes: the batch-formation hold for a lone fresh candidate and the post-retirement refill grace. Each was checked once per tick and skipped cold prefill for the whole tick while open.
- Main with everything plain, and the slot arm, never desynchronize their two clients. Both requests arrive together, are batch-primed, and no candidate is ever alone.

## The change

The batch-formation hold waits out the rest of its window inside the tick when that rest is shorter than the previous tick, and primes in this tick if nothing arrives in it. Otherwise it keeps main's skip. The refill grace keeps main's skip: its evidence (the mixed hit/cold c = 4 hit TTFT) was earned with it, and a replacement can still be crossing the HTTP layer when the window ends. `dedup_waiting` also keeps its skip. Both stay open in memra#1223.

## The fix against main (`gb10/ab/`, `gb10/ab-normal/`)

ABBA for the plain cells and the normal census: main2-21, grace2-22, grace2-23, main2-24. The default cells ran once per arm, after the ABBA (default-grace2, default-main2).

| boot | cell | long tokens | long ITL p50 / p99 ms | short n | short p50 / p95 s | plain primes: ticks to prime | waits (ms total) |
|---|---|---:|---|---:|---|---|---|
| plain-main2-21 | 1:200 | 310 | 75.0 / 185.0 | 62 | 0.283 / 0.286 | {2: 62} | 0 (0.0) |
| plain-main2-21 | 1:0 | 270 | 78.0 / 185.0 | 90 | 0.337 / 0.338 | {2: 90} | 0 (0.0) |
| plain-main2-21 | 2:200 | 270 | 79.0 / 220.0 | 136 | 0.246 / 0.249 | {1: 136} | 0 (0.0) |
| plain-main2-21 | 2:0 | 203 | 215.0 / 222.0 | 202 | 0.3 / 0.302 | {1: 202} | 0 (0.0) |
| plain-grace2-22 | 1:200 | 290 | 77.0 / 192.0 | 73 | 0.212 / 0.218 | {1: 73} | 74 (296.9) |
| plain-grace2-22 | 1:0 | 228 | 79.0 / 189.0 | 114 | 0.265 / 0.267 | {1: 114} | 115 (458.9) |
| plain-grace2-22 | 2:200 | 268 | 78.0 / 221.0 | 134 | 0.246 / 0.249 | {1: 134} | 1 (4.0) |
| plain-grace2-22 | 2:0 | 202 | 82.0 / 221.0 | 202 | 0.299 / 0.301 | {1: 202} | 1 (3.9) |
| plain-grace2-23 | 1:200 | 290 | 77.0 / 188.0 | 73 | 0.212 / 0.214 | {1: 73} | 74 (297.0) |
| plain-grace2-23 | 1:0 | 228 | 79.0 / 189.0 | 114 | 0.265 / 0.267 | {1: 114} | 115 (462.1) |
| plain-grace2-23 | 2:200 | 268 | 77.0 / 221.0 | 134 | 0.247 / 0.249 | {1: 134} | 1 (4.0) |
| plain-grace2-23 | 2:0 | 202 | 83.0 / 221.0 | 202 | 0.3 / 0.302 | {1: 202} | 1 (4.0) |
| default-grace2 | 1:0 | 229 | 0.0 / 316.0 | 97 | 0.309 / 0.315 | - | 0 (0.0) |
| default-grace2 | 1:200 | 424 | 0.0 / 314.0 | 62 | 0.308 / 0.311 | - | 0 (0.0) |
| default-grace2 | 1:1000 | 626 | 0.0 / 312.0 | 24 | 0.302 / 0.311 | - | 0 (0.0) |
| default-grace2 | 2:200 | 139 | 0.0 / 529.0 | 116 | 0.32 / 0.326 | - | 0 (0.0) |
| default-grace2 | 2:0 | 74 | 0.0 / 945.0 | 130 | 0.488 / 0.522 | - | 0 (0.0) |
| default-main2 | 1:0 | 230 | 0.0 / 318.0 | 98 | 0.309 / 0.318 | - | 0 (0.0) |
| default-main2 | 1:200 | 431 | 0.0 / 313.0 | 61 | 0.306 / 0.312 | - | 0 (0.0) |
| default-main2 | 1:1000 | 631 | 0.0 / 309.0 | 24 | 0.301 / 0.308 | - | 0 (0.0) |
| default-main2 | 2:200 | 139 | 0.0 / 529.0 | 117 | 0.319 / 0.326 | - | 0 (0.0) |
| default-main2 | 2:0 | 77 | 0.0 / 946.0 | 130 | 0.501 / 0.525 | - | 0 (0.0) |
| plain-main2-24 | 1:200 | 313 | 74.0 / 201.0 | 63 | 0.278 / 0.324 | {2: 63} | 0 (0.0) |
| plain-main2-24 | 1:0 | 270 | 77.0 / 185.0 | 90 | 0.334 / 0.336 | {2: 90} | 0 (0.0) |
| plain-main2-24 | 2:200 | 270 | 78.0 / 220.0 | 136 | 0.244 / 0.246 | {1: 136} | 0 (0.0) |
| plain-main2-24 | 2:0 | 202 | 82.0 / 222.0 | 202 | 0.299 / 0.301 | {1: 202} | 0 (0.0) |

| boot | c | agg tok/s | TTFT p50 s | ITL p50 / p99 ms | K at admission | texts equal to main |
|---|---:|---:|---:|---|---|---|
| main2-21 | 1 | 24.7 | 0.301 | 0.0 / 106.9 | K3/cost-model x3 | 3 of 3 |
| main2-21 | 2 | 39.93 | 0.462 | 0.0 / 229.0 | K3/cost-model x6 | 6 of 6 |
| main2-21 | 4 | 52.98 | 0.914 | 0.0 / 478.6 | K3/cost-model x12 | 12 of 12 |
| main2-21 | 8 | 79.95 | 1.521 | 87.5 / 102.4 | K0/cost-model x22 K1/resume-floor(k=0->1) x1 K3/cost-model x1 | 24 of 24 |
| grace2-22 | 1 | 24.68 | 0.298 | 0.0 / 106.9 | K3/cost-model x3 | 3 of 3 |
| grace2-22 | 2 | 35.93 | 0.302 | 0.0 / 240.2 | K3/cost-model x6 | 6 of 6 |
| grace2-22 | 4 | 53.05 | 0.908 | 0.0 / 478.6 | K3/cost-model x12 | 12 of 12 |
| grace2-22 | 8 | 79.86 | 1.416 | 87.4 / 102.2 | K0/cost-model x23 K3/cost-model x1 | 24 of 24 |
| grace2-23 | 1 | 24.68 | 0.298 | 0.0 / 105.8 | K3/cost-model x3 | 3 of 3 |
| grace2-23 | 2 | 39.97 | 0.459 | 0.0 / 228.5 | K3/cost-model x6 | 6 of 6 |
| grace2-23 | 4 | 53.81 | 0.917 | 0.0 / 421.6 | K3/cost-model x12 | 12 of 12 |
| grace2-23 | 8 | 79.77 | 1.464 | 87.4 / 102.4 | K0/cost-model x23 K3/cost-model x1 | 24 of 24 |
| main2-24 | 1 | 24.66 | 0.298 | 0.0 / 106.3 | K3/cost-model x3 | 3 of 3 |
| main2-24 | 2 | 40.02 | 0.457 | 0.0 / 229.3 | K3/cost-model x6 | 6 of 6 |
| main2-24 | 4 | 52.76 | 0.686 | 0.0 / 468.9 | K3/cost-model x12 | 12 of 12 |
| main2-24 | 8 | 79.85 | 1.429 | 87.4 / 102.3 | K0/cost-model x23 K3/cost-model x1 | 24 of 24 |

- **Plain one-client cells.** Main primes every lone plain request in its second tick (62 of 62, 90 of 90, 63 of 63, 90 of 90). The fix primes each in its first (73 of 73 and 114 of 114 in both boots). Short p50 falls by one tick: 0.278-0.283 to 0.212 s at 200 ms think, 0.334-0.337 to 0.265 s at 0 ms. The closed-loop client then sends 16 to 27% more requests in the 30 s, and the long stream's tokens fall by the prefill those extra requests bring (310-313 to 290, 270 to 228).
- **What the wait costs.** One wait per lone arrival: 74 and 115 per 30 s at one client (2.5 and 3.8 per second). Each waits its full window, about 4.0 ms, since one client never sends a second request inside it. That adds at most 4 ms to the tick it falls in, about 5% of a 75 ms decode step. On the long stream it shows as about +2 ms on ITL p50, with p99 inside main's spread.
- **Unchanged cells.** Two clients in lockstep (one wait per cell) and every default cell (the short requests go speculative: 0 waits) give main's rows. In the default cells the long stream's ITL p50 is 0 on both arms: its speculative bursts emit several tokens at once.
- **Ordinary traffic.** In the normal census the fix waited 0 times in every cell, so no lone plain arrival occurs there. K at admission and all 45 texts per boot are equal to main. One fix boot's c = 2 ran at 35.9 tok/s with TTFT p50 0.30 s, the second mode of c = 2's arrival grouping, which main also showed in the first-form census (35.84 tok/s, 0.301 s; those receipts are in this PR's history at b21d56b5b2).

## The c = 4 TTFT spread is main's (`gb10/c4/`)

Measured on the first form (81b70a7a8e, which also waited in the refill grace). Its census's c = 4 TTFT p50 came out 0.912 and 0.907 s on main and 2.419 and 2.353 s on the fix, with 0 waits in those cells. When no wait opens, the fix takes exactly main's decisions. Repeating the phase four times per boot (`PLAN=4,4,4,4`, ABBA: grace-5, main2-6, main2-7, grace-8) gives ten c = 4 starts per arm, counting the census's two (`tools/c4start.py`):

| how the four arrivals reach the worker | main | the fix | TTFT p50 s |
|---|---:|---:|---:|
| 1 then 3 | 6 | 7 | 0.91 |
| 2 then 2 | 3 | 3 | 2.16 to 2.29 |
| 3 then 1 | 1 | 0 | 0.69 |

The grouping follows arrival timing, on both arms. In a 2-then-2 start the first two sessions form a pooled MTP cohort, and its burst runs before the next two newcomers' primes, so their first token comes after about 1.2 s more. That is main's behaviour, filed as memra#1227. Every c = 4 phase of the repeats has 0 waits and the same draft length (K3) at every admission. Option A's own census shows the same spread: c = 4 TTFT p50 0.686 to 0.917 s on both arms.

The per-phase rows:

| boot | phase | first tick: live sessions | ticks until all 4 primed | TTFT p50 s | TTFTs sorted (first 4) |
|---|---:|---:|---:|---:|---|
| grace-5 | 0 | 1 | 2 | 0.908 | [0.241, 0.512, 0.682, 0.684] |
| grace-5 | 1 | 1 | 2 | 0.909 | [0.227, 0.46, 0.681, 0.683] |
| grace-5 | 2 | 2 | 2 | 2.280 | [0.23, 0.455, 1.682, 1.908] |
| grace-5 | 3 | 1 | 2 | 0.91 | [0.238, 0.471, 0.684, 0.685] |
| main2-6 | 0 | 2 | 2 | 2.182 | [0.254, 0.489, 1.584, 1.8] |
| main2-6 | 1 | 3 | 2 | 0.692 | [0.237, 0.459, 0.462, 0.465] |
| main2-6 | 2 | 1 | 2 | 0.908 | [0.228, 0.462, 0.68, 0.683] |
| main2-6 | 3 | 2 | 2 | 2.291 | [0.23, 0.457, 1.815, 1.979] |
| main2-7 | 0 | 1 | 2 | 0.908 | [0.245, 0.516, 0.683, 0.684] |
| main2-7 | 1 | 2 | 2 | 2.264 | [0.233, 0.459, 1.571, 1.79] |
| main2-7 | 2 | 1 | 2 | 0.911 | [0.229, 0.461, 0.683, 0.687] |
| main2-7 | 3 | 1 | 2 | 0.909 | [0.229, 0.462, 0.682, 0.682] |
| grace-8 | 0 | 1 | 2 | 0.906 | [0.246, 0.517, 0.68, 0.681] |
| grace-8 | 1 | 1 | 2 | 0.908 | [0.226, 0.467, 0.682, 0.682] |
| grace-8 | 2 | 1 | 2 | 0.908 | [0.227, 0.46, 0.682, 0.682] |
| grace-8 | 3 | 1 | 2 | 0.908 | [0.228, 0.461, 0.681, 0.684] |

## Files

- `tools/`: the cells, `primeticks.py`, `summgrace.py`, `c4start.py`, `abtables.py`.
- `gb10/diag/`: the diagnostic boots. `gb10/ab/` and `gb10/ab-normal/`: the A/B on 7c8028cd17. `gb10/c4/`: the c = 4 repeats on the first form.
- Each directory holds the rows (`stall.jsonl` or `normal.jsonl`), `guard.log` (binary hash, device, temperature), `boots.log` and the server logs with the tick trace.
