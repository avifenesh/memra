# The 4 ms prime holds wait inside the tick instead of skipping it (memra#1223)

Date: 2026-11-11. Found while measuring memra#1211.

## Conditions

- GB10 (one DGX Spark unit), Qwen3.8-27B NVFP4 + Q5_K with its MTP head, context 8,192, eight sessions, `MEMRA_TICK_TRACE=1` in every boot. Pass/fail style, single runs per boot, one boot per lease.
- Binaries, both built on the same unit:
  - `main2`: main ede1b8119f, `memra-0.141.1-0c5eb68559cb`.
  - `grace`: the fix 81b70a7a8e, `memra-0.141.1-2f54e998034a`.

  The PR head adds a docs commit only (FLAGS.md and a comment naming the issue).
- Cells (`tools/stall_cell.py`): a long greedy streamed request runs alone for 8 s. Then closed-loop clients send one-token chat requests for 30 s, each waiting the think time after every answer. Then the long request runs alone for 12 s. The rows carry the long stream's inter-token gaps (p50 and p99 inside the window) and the hold census from the trace. `tools/normal_cell.py` with `STREAM=1`: c = 1, 2, 4, 8 clients, three greedy 160-token streamed requests each, with TTFT and inter-token gaps.
- `tools/primeticks.py` counts, for every plain admission, the ticks from its admission to the tick that primes it. 1 means its admission tick.

## The diagnosis (`gb10/diag/`)

memra#1211's stall-mark form (dd243cf79c on 83d37f5179) admits one-token requests to plain decode beside a demoted long stream. At 2 clients and 200 ms think, its short p50 was 0.476 s against main's 0.318 s.

| boot | 2:200 long tokens / short p50 / p95 | plain primes, ticks to prime | 2:0 short p50 |
|---|---|---|---|
| #1211 form, default holds | 262 / 0.476 / 0.479 | 3rd tick for 85 of 86 | 0.301 |
| the same, `MEMRA_PRIME_BATCH_HOLD_MS=0` | 224 / 0.327 / 0.329 | 1st for 110 of 110 | 0.301 |
| main, `MEMRA_SPEC_K=0` | 270 / 0.246 / 0.249 | 1st for 136 of 136 | 0.299 |
| main, default (shorts speculative) | 142 / 0.318 / 0.324 | none plain | 0.458 |
| the #1201 slot-arm probe (2628ece940) | 267 / 0.248 / 0.252 | 1st for 134 of 134 | 0.301 |

- The two windows behind the 3rd-tick primes are the batch-formation hold for a lone fresh candidate and the post-retirement refill grace. Each was checked once per tick and skipped cold prefill for the whole tick while open.
- Main with everything plain, and the slot arm, never desynchronize their two clients. Both requests arrive together, are batch-primed, and no candidate is ever alone.

## The fix against main (`gb10/ab/`, `gb10/ab-normal/`)

ABBA: main2-1, grace-2, grace-3, main2-4.

| boot | cell | long tokens | long ITL p50 / p99 ms | short n | short p50 / p95 s | hold waits (ms total) |
|---|---|---:|---|---:|---|---|
| plain-main2-1 | 1:200 | 313 | 74.0 / 185.0 | 63 | 0.282 / 0.285 | 0 (0) |
| plain-main2-1 | 1:0 | 270 | 78.0 / 185.0 | 90 | 0.336 / 0.338 | 0 (0) |
| plain-main2-1 | 2:200 | 270 | 79.0 / 221.0 | 136 | 0.246 / 0.249 | 0 (0) |
| plain-main2-1 | 2:0 | 202 | 84.0 / 222.0 | 202 | 0.3 / 0.302 | 0 (0) |
| default-main2-1 | 1:0 | 230 | 0.0 / 317.0 | 98 | 0.309 / 0.314 | 0 (0) |
| default-main2-1 | 1:200 | 424 | 0.0 / 314.0 | 61 | 0.306 / 0.311 | 0 (0) |
| default-main2-1 | 1:1000 | 630 | 0.0 / 310.0 | 24 | 0.302 / 0.307 | 0 (0) |
| default-main2-1 | 2:200 | 139 | 0.0 / 526.0 | 117 | 0.318 / 0.323 | 0 (0) |
| default-main2-1 | 2:0 | 77 | 0.0 / 943.0 | 130 | 0.477 / 0.519 | 0 (0) |
| plain-grace-2 | 1:200 | 290 | 76.0 / 190.0 | 73 | 0.212 / 0.217 | 74 (297.04) |
| plain-grace-2 | 1:0 | 226 | 80.0 / 198.0 | 113 | 0.265 / 0.274 | 114 (457.39) |
| plain-grace-2 | 2:200 | 270 | 78.0 / 221.0 | 136 | 0.246 / 0.249 | 1 (4.0) |
| plain-grace-2 | 2:0 | 202 | 83.0 / 222.0 | 202 | 0.299 / 0.302 | 1 (4.0) |
| default-grace-2 | 1:0 | 230 | 0.0 / 317.0 | 98 | 0.31 / 0.313 | 0 (0.0) |
| default-grace-2 | 1:200 | 418 | 0.0 / 313.0 | 62 | 0.307 / 0.311 | 0 (0.0) |
| default-grace-2 | 1:1000 | 630 | 0.0 / 311.0 | 24 | 0.303 / 0.311 | 0 (0.0) |
| default-grace-2 | 2:200 | 139 | 0.0 / 529.0 | 117 | 0.318 / 0.328 | 0 (0.0) |
| default-grace-2 | 2:0 | 77 | 0.0 / 947.0 | 130 | 0.495 / 0.52 | 0 (0.0) |
| plain-grace-3 | 1:200 | 290 | 76.0 / 191.0 | 73 | 0.212 / 0.217 | 74 (297.05) |
| plain-grace-3 | 1:0 | 228 | 87.0 / 189.0 | 114 | 0.265 / 0.267 | 115 (462.03) |
| plain-grace-3 | 2:200 | 270 | 78.0 / 221.0 | 136 | 0.246 / 0.249 | 1 (4.0) |
| plain-grace-3 | 2:0 | 202 | 82.0 / 222.0 | 202 | 0.3 / 0.301 | 1 (3.99) |
| default-grace-3 | 1:0 | 228 | 0.0 / 318.0 | 97 | 0.31 / 0.314 | 0 (0.0) |
| default-grace-3 | 1:200 | 426 | 0.0 / 317.0 | 60 | 0.308 / 0.316 | 0 (0.0) |
| default-grace-3 | 1:1000 | 626 | 0.0 / 311.0 | 24 | 0.303 / 0.308 | 0 (0.0) |
| default-grace-3 | 2:200 | 143 | 0.0 / 531.0 | 116 | 0.32 / 0.327 | 0 (0.0) |
| default-grace-3 | 2:0 | 74 | 0.0 / 948.0 | 130 | 0.435 / 0.523 | 0 (0.0) |
| plain-main2-4 | 1:200 | 310 | 75.0 / 185.0 | 62 | 0.283 / 0.287 | 0 (0) |
| plain-main2-4 | 1:0 | 270 | 78.0 / 185.0 | 90 | 0.337 / 0.338 | 0 (0) |
| plain-main2-4 | 2:200 | 270 | 79.0 / 221.0 | 136 | 0.246 / 0.248 | 0 (0) |
| plain-main2-4 | 2:0 | 202 | 82.0 / 222.0 | 202 | 0.3 / 0.302 | 0 (0) |
| default-main2-4 | 1:0 | 229 | 0.0 / 319.0 | 97 | 0.31 / 0.313 | 0 (0) |
| default-main2-4 | 1:200 | 427 | 0.0 / 313.0 | 60 | 0.308 / 0.316 | 0 (0) |
| default-main2-4 | 1:1000 | 630 | 0.0 / 311.0 | 24 | 0.303 / 0.309 | 0 (0) |
| default-main2-4 | 2:200 | 139 | 0.0 / 531.0 | 116 | 0.32 / 0.328 | 0 (0) |
| default-main2-4 | 2:0 | 74 | 0.0 / 947.0 | 130 | 0.436 / 0.52 | 0 (0) |

| boot | c | agg tok/s | TTFT p50 s | ITL p50 / p99 ms | K at admission | hold waits (ms total) | texts |
|---|---:|---:|---:|---|---|---|---|
| main2-1 | 1 | 24.66 | 0.301 | 0.0 / 107.2 | K3/cost-model x3 | 0 (0) | 3 of 3 equal to main |
| main2-1 | 2 | 35.84 | 0.301 | 0.0 / 238.7 | K3/cost-model x6 | 0 (0) | 6 of 6 equal to main |
| main2-1 | 4 | 53.18 | 0.912 | 0.0 / 446.0 | K3/cost-model x12 | 0 (0) | 12 of 12 equal to main |
| main2-1 | 8 | 80.08 | 1.28 | 87.6 / 102.1 | K0/cost-model x22 K1/resume-floor(k=0->1) x1 K3/cost-model x1 | 0 (0) | 24 of 24 equal to main |
| grace-2 | 1 | 24.77 | 0.3 | 0.0 / 106.5 | K3/cost-model x3 | 0 (0.0) | 3 of 3 equal to main |
| grace-2 | 2 | 35.9 | 0.301 | 0.0 / 239.3 | K3/cost-model x6 | 0 (0.0) | 6 of 6 equal to main |
| grace-2 | 4 | 53.08 | 2.419 | 0.0 / 379.0 | K3/cost-model x12 | 0 (0.0) | 12 of 12 equal to main |
| grace-2 | 8 | 79.64 | 1.455 | 87.5 / 102.6 | K0/cost-model x23 K3/cost-model x1 | 0 (0.0) | 24 of 24 equal to main |
| grace-3 | 1 | 24.66 | 0.298 | 0.0 / 106.4 | K3/cost-model x3 | 0 (0.0) | 3 of 3 equal to main |
| grace-3 | 2 | 40.01 | 0.461 | 0.0 / 229.0 | K3/cost-model x6 | 0 (0.0) | 6 of 6 equal to main |
| grace-3 | 4 | 52.85 | 2.353 | 0.0 / 370.6 | K3/cost-model x12 | 0 (0.0) | 12 of 12 equal to main |
| grace-3 | 8 | 79.77 | 1.445 | 87.5 / 102.4 | K0/cost-model x23 K3/cost-model x1 | 0 (0.0) | 24 of 24 equal to main |
| main2-4 | 1 | 24.63 | 0.299 | 0.0 / 107.2 | K3/cost-model x3 | 0 (0) | 3 of 3 equal to main |
| main2-4 | 2 | 40.07 | 0.458 | 0.0 / 228.5 | K3/cost-model x6 | 0 (0) | 6 of 6 equal to main |
| main2-4 | 4 | 53.37 | 0.907 | 0.0 / 444.0 | K3/cost-model x12 | 0 (0) | 12 of 12 equal to main |
| main2-4 | 8 | 79.78 | 1.445 | 87.4 / 102.2 | K0/cost-model x23 K3/cost-model x1 | 0 (0) | 24 of 24 equal to main |

- **Plain one-client cells.** Main primes every lone plain request in its second tick (63 of 63, 90 of 90, 62 of 62, 90 of 90). The fix primes each in its first (73 of 73, 113 of 113, 73 of 73, 114 of 114). Short p50 falls by one tick: 0.282 to 0.212 s at 200 ms think, 0.336 to 0.265 s at 0 ms. The closed-loop clients then send 16 to 27% more requests in the 30 s, and the long stream's tokens fall by the prefill those extra requests bring (313 to 290, 270 to 227).
- **What the wait costs.** One wait per lone arrival, 74 and 114 to 115 per 30 s at one client. Each waits its full 4 ms, since one client never sends a second request inside the window. That adds at most 4 ms to the tick it falls in, about 5% of a 75 ms decode step. On the long stream it shows as 2 to 9 ms on ITL p50 and 4 to 13 ms on p99.
- **Unchanged cells.** Two clients in lockstep (one wait per cell) and every default cell (the short requests go speculative: 0 waits) give the same rows as main.
- **Ordinary traffic.** In the normal census the fix waited 0 times in every cell. No lone plain arrival occurs there: the c = 8 plain admissions arrive in waves and are primed in their admission tick on both arms. K at admission and all 45 texts per boot are equal to main.

## The c = 4 TTFT spread is main's (`gb10/c4/`)

The census's c = 4 TTFT p50 came out 0.912 and 0.907 s on main and 2.419 and 2.353 s on the fix, with 0 waits in those cells. When no wait opens, the fix takes exactly main's decisions. Repeating the phase four times per boot (`PLAN=4,4,4,4`, ABBA: grace-5, main2-6, main2-7, grace-8) gives ten c = 4 starts per arm, counting the census's two (`tools/c4start.py`):

| how the four arrivals reach the worker | main | the fix | TTFT p50 s |
|---|---:|---:|---:|
| 1 then 3 | 6 | 7 | 0.91 |
| 2 then 2 | 3 | 3 | 2.16 to 2.29 |
| 3 then 1 | 1 | 0 | 0.69 |

The grouping follows arrival timing, on both arms. In a 2-then-2 start the first two sessions form a pooled MTP cohort, and its burst runs before the next two newcomers' primes, so their first token comes after about 1.2 s more. That is main's behaviour, filed as memra#1227. Every c = 4 phase of the repeats has 0 hold waits and the same draft length (K3) at every admission.

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

- `tools/`: the cells, `primeticks.py`, `summgrace.py`, `c4start.py`.
- `gb10/diag/`: the diagnostic boots. `gb10/ab/` and `gb10/ab-normal/`: the A/B. `gb10/c4/`: the c = 4 repeats.
- Each directory holds the rows (`stall.jsonl` or `normal.jsonl`), `guard.log` (binary hash, device, temperature), `boots.log` and the server logs with the tick trace.
